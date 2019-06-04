import argparse
import imageio
from math import inf
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import os
import pickle
from ripser import ripser, Rips
from persim import plot_diagrams


def generate(i, all_block_list, names=[]):
    bs = (0, 0)
    block_lists = []
    all_images = []
    directory = "images/batches/{0}".format(i)
    for j, block in enumerate(all_block_list):
        if block.shape != bs:
            block_lists.append([])
            bs = block.shape
        block_lists[-1].append(({"block": block, "name": names[j]}))
    for j, block_list in enumerate(block_lists):
        all_images += generate_diagrams('{0}'.format(directory
                                                     ),
                                        [block["block"]
                                            for block in block_list],
                                        names=[block["name"]
                                               for block in block_list],
                                        c=len(all_images))
    # visualize(all_images)
    # imageio.mimsave('{0}/all.gif'.format(directory), all_images, fps=3)
    return all_images


def generate_diagrams(dn, block_list, names=[], c=0):
    """
    Given layers of batch data, generate their Vietoris-Rips Complexes and persistence diagrams
    :params i:
    :params block: list of second-order numpy array
    """
    directory = dn
    os.makedirs(directory, exist_ok=True)  # succeeds even if directory exists.
    os.makedirs(directory + "/pickle_data", exist_ok=True)
    images = []
    diagrams_list = []

    birth = 0
    death = 0

    bs = (0, 0)

    for j, block in enumerate(block_list):
        rips = Rips(verbose=False)
        diagrams = rips.fit_transform(block)
        for k, diagram in enumerate(diagrams):
            if len(diagram) != 0:
                birth = max(birth, max(diagram[:, 0]))
                death = max(death, max([v for v in diagram[:, 1] if v != inf]))
        diagrams_list.append(diagrams)

    birth = int(birth)
    death = int(death)

    for j, diagrams in enumerate(diagrams_list):
        label = str(c) + "_" + names[j].replace("/", "_")
        fn = '{0}/{1}'.format(directory, label)
        c += 1

        with open("{directory}/pickle_data/{label}.pickle".format(directory=directory, label=label), 'wb') as f:
            pickle.dump(diagrams, f)
        fn += ".png"

        images.append(fn)
    return images

    """

    # rips.plot(diagrams)
    fig = plt.figure()
    for k, diagram in enumerate(diagrams):
        plt.scatter(diagram[:, 0], diagram[:, 1])

    plt.title(label)
    axes = plt.gca()
    axes.set_xlim([0, birth + 10])
    axes.set_ylim([0, death + 10])

    plt.savefig(fn)
    plt.clf()
    images.append(imageio.imread(fn))
    imageio.mimsave('{0}/all.gif'.format(directory), images, fps=3)
    """

def gen_pd(dir_layers, layer_no):
    dir_layers += '/layer_'+str(layer_no)
    # Read pickle data
    layer = []
    with open('{dir_layers}/all.pickle'.format(dir_layers=dir_layers, layer_no=layer_no), 'rb') as f:
        layer = np.vstack(pickle.load(f))
    # Generate Vietoris-Rips Complex
    rips = Rips(verbose=False)
    diagrams = rips.fit_transform(layer)
    fig = plt.figure()
    # Save persistence diagrams as image
    for k, diagram in enumerate(diagrams):
        plt.scatter(diagram[:,0], diagram[:,1])
        for r, row in enumerate(diagram):
            diagram[r] = np.array([(row[0]+row[1])/sqrt(2), (row[1]-row[0])/sqrt(2)]) # rotate entries 45 degrees
    plt.title("Layer {0}".format(layer_no))
    plt.savefig("images/layer_{layer_no}.png".format(layer_no=layer_no))
    plt.clf()

    # Save rotated persistence diagrams as image
    for k, diagram in enumerate(diagrams):
        plt.scatter(diagram[:,0], diagram[:,1])
    plt.title("Layer 5 Landscape")
    plt.savefig("{landscape_dir}/layer_{layer_no}.png".format(landscape_dir=landscape_dir, layer_no=layer_no))
    plt.clf()
    print("done")

if __name__ == "__main__":
    # Parse aruments
    parser = argparse.ArgumentParser('Create persistence diagram')
    g = parser.add_argument_group('computation parameters')
    g.add_argument('--dir', default='pickle_data')
    g.add_argument('--layer_no', type=int, default=0)
    args = parser.parse_args()

    landscape_dir = 'images/landscape'
    os.makedirs(landscape_dir, exist_ok=True)

    gen_pd(args.dir, args.layer_no)
