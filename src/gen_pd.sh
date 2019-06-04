#!/bin/bash

for i in {0..417}
do
    echo $i
    python -u mkProjections.py --layer_no $i
    python -u batch_transform.py --layer_no $i
    python -u viz.py --layer_no $i

    rm -rf pickle_data/layer*
done

