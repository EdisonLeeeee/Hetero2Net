import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from functools import partial
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
from copy import copy
from collections import defaultdict
from texttable import Texttable
from torch.cuda.amp import GradScaler, autocast

from hetero2net.metapath import drop_metapath

def train_mini_batch(model, optimizer, train_loader, device, p=1.,
                     metapaths=None, r=0.5, alpha=0.2, beta=0.5, schedule=None):
    model.train()
    amp_emabled = device.type=='cuda'
    scaler = GradScaler(enabled=amp_emabled)
    
    loss_all = 0.
    num_batches = 0
    node_type = model.node_type
    
    for batch in tqdm(train_loader):
        batch = batch.to(device, 'x', 'edge_index')
        if metapaths is not None:
            batch = drop_metapath(batch, metapaths, r=r)
        batch_size = batch[node_type].batch_size
        
        y_emb = batch[node_type].get('y_emb')
        if y_emb is not None:
            y_emb = y_emb.clone()
            ratio = p
            if ratio < 1:
                n = batch_size                
                index = torch.arange(n)[torch.rand(n) < ratio]            
                y_emb[index] = model.out_channels # mask current batch nodes    
            else:
                y_emb[:batch_size] = model.out_channels # mask current batch nodes    
                
        # with autocast(enabled=amp_emabled):
        y_pred, homos, heteros = model(batch.x_dict, batch.edge_index_dict, y_emb)
        if isinstance(y_pred, dict):
            y_pred = y_pred[node_type]
        y_pred = y_pred[:batch_size]
        y_true = batch[node_type].y[:batch_size]
        if y_true.ndim > 1:
            loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        else:
            loss = F.cross_entropy(y_pred, y_true)
    
        #############################################
        if metapaths is not None and beta > 0:
            out_dict1 = defaultdict(list)
            out_dict2 = defaultdict(list)
            for homo_dict, hetero_dict in zip(homos, heteros):
                for edge_type in homo_dict.keys():
                    out_dict1[edge_type].append(homo_dict[edge_type])
                    out_dict2[edge_type].append(hetero_dict[edge_type])

            z_dict_homo = {}
            z_dict_hetero = {}
            for edge_type in out_dict1.keys():
                src, rel, dst = edge_type
                z_dict_homo[(src, dst)] = torch.cat(out_dict1[edge_type], dim=1)
                z_dict_hetero[(src, dst)] = torch.cat(out_dict2[edge_type], dim=1)
            del out_dict1, out_dict2

            for edge_type, pos_edge_index in batch.metapath_dict.items():
                src, dst = edge_type
                row = torch.randint(0, batch[src].x.size(0), size=(pos_edge_index.size(1),))
                col = torch.randint(0, batch[dst].x.size(0), size=(pos_edge_index.size(1),))
                neg_edge_index = torch.stack([row, col], dim=0).to(pos_edge_index)

                # with autocast(enabled=amp_emabled):
                link_pred_homo = model.edge_decoder(z_dict_homo[(dst, src)], 
                                                    z_dict_homo[(src, dst)], 
                                                    pos_edge_index)
                link_pred_hetero = model.edge_decoder(z_dict_hetero[(dst, src)], 
                                                      z_dict_hetero[(src, dst)], 
                                                      neg_edge_index)            

                loss_link = F.binary_cross_entropy_with_logits(link_pred_homo, torch.ones_like(link_pred_homo))
                loss_link += F.binary_cross_entropy_with_logits(link_pred_hetero, torch.ones_like(link_pred_hetero))
                loss += beta * loss_link         
        #############################################
        loss_all += loss.item()
        num_batches += 1
        scaler.scale(loss).backward()  
        scaler.step(optimizer)   
        scaler.update()   
        # loss.backward()
        # optimizer.step()   
        optimizer.zero_grad()
        if schedule is not None:
            schedule.step()        
    return loss_all / num_batches

@torch.no_grad()
def evaluate_mini_batch(model, test_loader, device, metrics):
    model.eval()
    preds = []
    labels = []
    node_type = model.node_type
    torch.cuda.empty_cache()
    amp_emabled = device.type=='cuda'
    
    for batch in tqdm(test_loader):
        batch = batch.to(device, 'x', 'edge_index')
        batch_size = batch[node_type].batch_size
        y_emb = batch[node_type].get('y_emb')
        with autocast(enabled=amp_emabled):
            y_pred = model(batch.x_dict, batch.edge_index_dict, y_emb)
            
        y_pred = y_pred.float()
        if isinstance(y_pred, dict):
            y_pred = y_pred[node_type]
        y_pred = y_pred[:batch_size]
        y_true = batch[node_type].y[:batch_size]
        preds.append(y_pred)
        labels.append(y_true)
        
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    if labels.ndim > 1:
        loss = F.binary_cross_entropy_with_logits(preds, labels)
    else:
        loss = F.cross_entropy(preds, labels)    
    return loss.item(), [evaluate(labels, preds, metric=metric) for metric in metrics]

def train_full_batch(model, optimizer, data, mask, device, p=0.7,
                     metapaths=None, r=0.5, alpha=0.2, beta=0.5, schedule=None):
    model.train()
    
    node_type = model.node_type
    data = data.to(device, 'edge_index')
    
    optimizer.zero_grad()
    y_emb = data[node_type].get('y_emb')
    if y_emb is not None:
        y_emb = y_emb.clone().to(device)
        ratio = p
        n = int(mask.sum())
        out = mask.clone()
        out[mask] = torch.rand(n, device=mask.device) < ratio
        y_emb[out] = model.out_channels # mask current train nodes       
        
    if metapaths is not None:
        data = drop_metapath(copy(data), metapaths, r=r)
        
    y_pred, homos, heteros = model(data.x_dict, data.edge_index_dict, y_emb)
    if isinstance(y_pred, dict):
        y_pred = y_pred[node_type]  
    y_pred = y_pred[mask]
    y_true = data[node_type].y[mask]
    if y_true.ndim > 1:
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
    else:
        loss = F.cross_entropy(y_pred, y_true)
    
    #############################################
    if alpha > 0:
        for homo_dict, hetero_dict in zip(homos, heteros):
            for edge_type in homo_dict.keys():
                x_homo = homo_dict[edge_type]
                x_hetero = hetero_dict[edge_type]
                loss += alpha * dist_corr(x_homo, x_hetero)
    #############################################
    
    #############################################
    if metapaths is not None and beta > 0:
        out_dict1 = defaultdict(list)
        out_dict2 = defaultdict(list)
        for homo_dict, hetero_dict in zip(homos, heteros):
            for edge_type in homo_dict.keys():
                out_dict1[edge_type].append(homo_dict[edge_type])
                out_dict2[edge_type].append(hetero_dict[edge_type])

        z_dict_homo = {}
        z_dict_hetero = {}
        for edge_type in out_dict1.keys():
            src, rel, dst = edge_type
            z_dict_homo[(src, dst)] = torch.cat(out_dict1[edge_type], dim=1)
            z_dict_hetero[(src, dst)] = torch.cat(out_dict2[edge_type], dim=1)
        del out_dict1, out_dict2
        
        for edge_type, pos_edge_index in data.metapath_dict.items():
            src, dst = edge_type
            row = torch.randint(0, data[src].x.size(0), size=(pos_edge_index.size(1),))
            col = torch.randint(0, data[dst].x.size(0), size=(pos_edge_index.size(1),))
            neg_edge_index = torch.stack([row, col], dim=0).to(pos_edge_index)
            
            link_pred_homo = model.edge_decoder(z_dict_homo[(dst, src)], 
                                                z_dict_homo[(src, dst)], 
                                                pos_edge_index)
            link_pred_hetero = model.edge_decoder(z_dict_hetero[(dst, src)], 
                                                  z_dict_hetero[(src, dst)], 
                                                  neg_edge_index)            
            
            loss_link = F.binary_cross_entropy_with_logits(link_pred_homo, torch.ones_like(link_pred_homo))
            loss_link += F.binary_cross_entropy_with_logits(link_pred_hetero, torch.ones_like(link_pred_hetero))
            loss += beta * loss_link       
    #############################################
    
    loss.backward()
    optimizer.step()  
    if schedule is not None:
        schedule.step()
    return loss.item()

@torch.no_grad()
def evaluate_full_batch(model, data, mask, device, metrics):
    model.eval()
    preds = []
    labels = []
    node_type = model.node_type
    data = data.to(device, 'edge_index')
    
    y_emb = data[node_type].get('y_emb')
    preds = model(data.x_dict, data.edge_index_dict, y_emb)
    if isinstance(preds, dict):
        preds = preds[node_type]    
    preds = preds[mask]
    labels = data[node_type].y[mask]
    if labels.ndim > 1:
        loss = F.binary_cross_entropy_with_logits(preds, labels)
    else:
        loss = F.cross_entropy(preds, labels)      
    return loss.item(), [evaluate(labels, preds, metric=metric) for metric in metrics]


def evaluate(y_true, y_pred, metric='acc'):
    if metric in ['ap', 'auc']:
        y_pred = F.softmax(y_pred, dim=1)[:, 1]
    else:
        if y_true.squeeze().ndim == 1:
            y_pred = y_pred.argmax(-1)
        else:
            # multi-classes
            y_pred = (y_pred > 0).float()
        
    if metric == 'acc':
        metric_fn = accuracy_score
    elif metric == 'micro-f1':
        metric_fn = partial(f1_score, average='micro')
    elif metric == 'macro-f1':
        metric_fn = partial(f1_score, average='macro')    
    elif metric == 'ap':
        metric_fn = average_precision_score
    elif metric == 'auc':
        metric_fn = roc_auc_score        
    else:
        raise ValueError(metric)  
        
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return metric_fn(y_true, y_pred)

def dist_corr(x1, x2):
    # Subtract the mean
    x1_mean = torch.mean(x1, dim=0, keepdim=True)
    x1 = x1 - x1_mean
    x2_mean = torch.mean(x2, dim=0, keepdim=True)
    x2 = x2 - x2_mean

    # Compute the cross correlation
    sigma1 = torch.sqrt(torch.mean(x1.pow(2)))
    sigma2 = torch.sqrt(torch.mean(x2.pow(2)))
    corr = torch.abs(torch.mean(x1*x2))/(sigma1*sigma2+1e-8)

    return corr

def auc_loss(pos_out, neg_out):
    return torch.square(1 - (pos_out - neg_out)).sum()

def hinge_auc_loss(pos_out, neg_out):
    return (torch.square(torch.clamp(1 - (pos_out - neg_out), min=0))).sum()

def log_rank_loss(pos_out, neg_out):
    return -torch.log(torch.sigmoid(pos_out - neg_out) + 1e-15).mean()

def ce_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss + neg_loss

def info_nce_loss(pos_out, neg_out):
    pos_exp = torch.exp(pos_out)
    neg_exp = torch.sum(torch.exp(neg_out), 1, keepdim=True)
    return -torch.log(pos_exp / (pos_exp + neg_exp) + 1e-15).mean()

def tab_printer(args):
    """Function to print the logs in a nice tabular format.
    
    Note
    ----
    Package `Texttable` is required.
    Run `pip install Texttable` if was not installed.
    
    Parameters
    ----------
    args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," "), args[k]] for k in keys])
    return t.draw()
