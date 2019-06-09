#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

# Delete scratch
rm -rf images/* && rm -rf pickle_data/layer* 
tree -I "projections|tensornets|__pycache__|*.pyc"

let batchsize=10
let num_batches=5000/$batchsize

for ((i=0; i<=$num_batches; i++)); do
    echo "Layer $i on" $(date)
    python -u mkProjections.py --layer_no $i --batchsize $batchsize &&
    python -u batch_transform.py --layer_no $i && 
    python -u viz.py --layer_no $i &&
    rm -rf pickle_data/layer*
done

# rm nohup.out
