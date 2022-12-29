python ./train.py --datapath "D:/FYP/Code/DCAMA/datasets" \
           --benchmark serp \
           --fold 0 \
           --bsz 1 \
           --nworker 0 \
           --backbone resnet50 \
           --logpath "./logs" \
           --lr 1e-3 \
           --nepoch 10