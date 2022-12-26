python ./train.py --datapath "/content/DCAMA/datasets" \
           --benchmark serp \
           --fold 0 \
           --bsz 4 \
           --nworker 0 \
           --backbone resnet50 \
           --logpath "./logs" \
           --lr 1e-3 \
           --nepoch 10