#! /bin/python

import viz
import datetime
import tables
import numpy as np
import cv2
import pickle
import tensornets as nets
import tensorflow as tf
import argparse
# from __future__ import print_function

########################################
print('processing cmd line args')

parser = argparse.ArgumentParser('generate layer projections')

g1 = parser.add_argument_group('computation options')
g1.add_argument('--batchsize', type=int, default=64)

g2 = parser.add_argument_group('output options')
g2.add_argument('--output_dir', type=str, default='projections')
g2.add_argument('--model', type=str, default='ResNet50', choices=['ResNet50'])
g2.add_argument('--data', type=str, default='cifar10', choices=['cifar10'])

args = parser.parse_args()

########################################


def filter_photo(Xtrain, Ytrain, class_no=0):
    """
    Filter photo by class number
    """

    x_train_filt = Xtrain[Ytrain.reshape(-1,) == class_no]
    y_train_filt = Ytrain[Ytrain.reshape(-1,) == class_no]

    return x_train_filt, y_train_filt


(Xtrain, Ytrain), (Xtest, Ytest) = tf.keras.datasets.cifar10.load_data()

Xtrain, Ytrain = filter_photo(Xtrain, Ytrain)

inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
model = nets.ResNet50(inputs, is_training=False)
load_model_weights = model.pretrained()

########################################


# filename = args.output_dir + '/' + args.model + '-' + args.data + '.h5'
# f = tables.open_file(filename, mode='w')
# atom = tables.Float64Atom()
# 
# 
# def sanitize_tensor_name(t):
#     return t.replace('/', '_').split(':')[0]
# 
# 
# arrays = [f.create_earray(
#     f.root,
#     sanitize_tensor_name(t.name),
#     atom,
#     [0] + t.get_shape().as_list()[1:]
# )
#     for t in model.get_middles()
# ]

########################################


def process_block(data):

    """
    Takes block data and reshapes it
    for Ripser

    :params data: middles from the tf.session graph execution
    :returns: numpy array representing (batch, (data, data, features))
    """

    blocks_processed = []
    for ix, single_block in enumerate(data):
        block_flattened = []
        for each_ibatch in single_block:
            flat_block = each_ibatch.flatten()
            block_flattened.append(flat_block)

        block_flattened = np.array(block_flattened)

        blocks_processed.append(block_flattened)

    return blocks_processed


def default_original():
    tf.get_default_graph().finalize()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(load_model_weights)

        max_data = Xtrain.shape[0]
        # while stop<max_data:
        for i in range(0, int(max_data / args.batchsize + 1)):
            print('' + str(datetime.datetime.now()) + ': i=', i)
            start = i * args.batchsize
            stop = min((i + 1) * args.batchsize, max_data)
            img = Xtrain[start:stop, :, :, :]

            img_resize = np.zeros([args.batchsize, 224, 224, 3])
            for j in range(start, stop):
                j -= start
                img_resize[j, :, :, 0] = cv2.resize(img[j, :, :, 0], (224, 224))
                img_resize[j, :, :, 1] = cv2.resize(img[j, :, :, 1], (224, 224))
                img_resize[j, :, :, 2] = cv2.resize(img[j, :, :, 2], (224, 224))
            img_resize = model.preprocess(img_resize)  # ,is_training=False)
            # middles=sess.run(model.get_middles(), {inputs: img_resize})
            all_middles = sess.run(model.get_all(), {inputs: img_resize})
            ph = open("tensor_names.pickle", "rb")
            all_names = pickle.load(ph)
            ph.close()
            all_block_list = process_block(all_middles)
            viz.generate(i, all_block_list, names=all_names)

            print(i)


def single_batch_run():

    """
    Single Batch Run
    """

    tf.get_default_graph().finalize()
    with tf.Session() as sess:
        print("Is there GPU being used?")
        results = sess.list_devices()
        print("results: ", results)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(load_model_weights)

        max_data = Xtrain.shape[0]
        # while stop<max_data:

        args.batchsize = 64 # max_data // 10
        print(args.batchsize)

        for i in range(0, int(max_data / args.batchsize + 1)):
            print('' + str(datetime.datetime.now()) + ': i=', i)
            start = i * args.batchsize
            stop = min((i + 1) * args.batchsize, max_data)
            img = Xtrain[start:stop, :, :, :]

            img_resize = np.zeros([args.batchsize, 224, 224, 3])

            for j in range(start, stop):
                j -= start
                img_resize[j, :, :, 0] = cv2.resize(img[j, :, :, 0], (224, 224))
                img_resize[j, :, :, 1] = cv2.resize(img[j, :, :, 1], (224, 224))
                img_resize[j, :, :, 2] = cv2.resize(img[j, :, :, 2], (224, 224))

            print("len(img_resize): ", len(img_resize))
            img_resize = model.preprocess(img_resize)  # ,is_training=False)
            print("running session")
            all_middles = sess.run(model.get_all(), {inputs: img_resize})

            # ph = open("pickle_data/batch_{0}.pickle".format(i), "wb")
            # pickle.dumps(all_middles, ph)
            # ph.close()

            print("got all_middles")
            ph = open("pickle_data/tensor_names.pickle", "rb")
            all_names = pickle.load(ph)
            ph.close()

            all_block_list = process_block(all_middles)
            print("visuals")
            viz.generate(i, all_block_list, names=all_names)

            print(i)
            exit()


if __name__ == '__main__':
    single_batch_run()
