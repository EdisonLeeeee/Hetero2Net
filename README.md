# Hetero $^2$ Net
Implementation of Hetero $^2$ Net from the "Hetero $^2$ Net: Heterophily-aware Representation Learning on Heterogenerous Graphs" paper.


# Requirements
+ torch                    2.0.1
+ torch_geometric          2.4.0
+ torch-cluster            1.6.1+pt20cu117
+ torch-scatter            2.1.1+pt20cu117
+ torch-sparse             0.6.17+pt20cu117
+ torch-spline-conv        1.2.2+pt20cu117
+ texttable                1.6.7
+ termcolor                1.1.0
+ scikit-learn             1.0.2
+ scipy                    1.10.1
+ numpy                    1.22.4
+ tqdm                     4.64.0
+ CUDA 11.7

# Reproduction

## DBLP
python main.py --dataset DBLP --dropout 0.5 --lr 0.001 --beta 0.2 --alpha 0. --mask_lp --p 0.7

## IMDB
python main.py --dataset IMDB--dropout 0.9 --lr 0.005  --alpha 0.1 --mask_lp --p 0.6

## ACM
python main.py --dataset ACM --epochs 100 --mask_lp --p 0.7

## OGB-MAG
python main.py --dataset MAG --num_neighbors 15 15 --hidden 256 --beta 0.5 --mask_lp --p 1.0

## RCDD
python main.py --dataset RCDD --metrics ap micro-f1 macro-f1 --dropout 0.5 --epochs 100 --num_neighbors 15 15 --hidden 256 --beta 0.2 --mask_lp


