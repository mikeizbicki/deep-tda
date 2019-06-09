#!/usr/bin/env python3

import argparse
import numpy as np
import os
import pickle
import sys
from tqdm import tqdm

def transform_batch_to_single(dir_layers, layer_no, split_str=None):
    """
    Load layer files and transforms them to single
    pickle
    """
    dir_layers = dir_layers+'/layer_{layer_no}'.format(layer_no=layer_no)

    pickle_files = os.listdir(dir_layers)

    if split_str is None:
        split_str = 'batch_'.format(layer_no=layer_no)

    files = [(x, int(x.split(split_str)[1].split('.pickle')[0])) for x in pickle_files]

    sorted_files = [x[0] for x in sorted(files, key=lambda x: x[1])]
    print(len(files), len(sorted_files))
    final_array = []

    for b, batch_name in enumerate(tqdm(sorted_files)):
        batch_file = os.path.join(dir_layers, batch_name)
        with open(batch_file, 'rb') as f:
            pickle_data = pickle.load(f)
            if b is 0:
                final_array = pickle_data
            final_array = np.vstack((final_array,pickle_data))

    with open('{dir_layers}/all.pickle'.format(dir_layers=dir_layers), 'wb') as ph:
        pickle.dump(final_array, ph)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Transform Pickles into Single Batch')
    g1 = parser.add_argument_group('computation parameters')
    g1.add_argument('--layer_no', type=int, default=0)
    g1.add_argument('--dir', default='pickle_data')

    args = parser.parse_args()

    transform_batch_to_single(args.dir, args.layer_no)
    print("Batch Transform Complete Layer - {layer}".format(layer=args.layer_no))
