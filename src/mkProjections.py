#!../venv/bin/python3

import datetime
import tables
import os
import numpy as np
import cv2
import pickle
import tensornets as nets
import tensorflow as tf
import argparse
from tqdm import tqdm

# from __future__ import print_function


def filter_photo(Xtrain, Ytrain, class_no=0):
    """
    Filter photo by class number
    """

    x_train_filt = Xtrain[Ytrain.reshape(-1,) == class_no]
    y_train_filt = Ytrain[Ytrain.reshape(-1,) == class_no]

    return x_train_filt, y_train_filt


def single_layer_run(layer_no, batch_size):

    """
    Single Batch Run
    """

    tf.get_default_graph().finalize()
    with tf.Session() as sess:
        results = sess.list_devices()
        print("GPU?: ", results)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(load_model_weights)


        for i in tqdm(range(0, int(max_data / batch_size + 1))):
            print('' + str(datetime.datetime.now()) + ': i=', i)
            start = i * batch_size
            stop = min((i + 1) * batch_size, max_data)

            img = Xtrain[start:stop, :, :, :]

            img_resize = np.zeros([batch_size, 224, 224, 3])

            for j in range(start, stop):
                j -= start

                img_resize[j, :, :, 0] = cv2.resize(
                    img[j, :, :, 0], (224, 224))
                img_resize[j, :, :, 1] = cv2.resize(
                    img[j, :, :, 1], (224, 224))
                img_resize[j, :, :, 2] = cv2.resize(
                    img[j, :, :, 2], (224, 224))

            img_resize = model.preprocess(img_resize)  # ,is_training=False)

            all_middles = sess.run(model.get_all(lay_no=layer_no), {inputs: img_resize})

            all_middles = np.array([x.flatten() for x in np.array(all_middles)])

            # Creates a dump of all the pickle for each batch
            with open('{dir_str}/batch_{batch_no}.pickle'.format(dir_str=dir_str, batch_no=i), 'wb') as f:
                pickle.dump(all_middles, f)


if __name__ == '__main__':
    print('processing cmd line args')
    parser = argparse.ArgumentParser('generate layer projections')
    g1 = parser.add_argument_group('computation options')
    g1.add_argument('--layer_no', type=int, default=0)
    g1.add_argument('--batchsize', type=int, default=12)

    args = parser.parse_args()

    dir_str = 'pickle_data/layer_{layer_no}'.format(layer_no=args.layer_no)
    directory = os.makedirs(dir_str, exist_ok=True)

    (Xtrain, Ytrain), (Xtest, Ytest) = tf.keras.datasets.cifar10.load_data()
    Xtrain, Ytrain = filter_photo(Xtrain, Ytrain)
    max_data = Xtrain.shape[0]

    inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    model = nets.ResNet50(inputs, is_training=False)
    load_model_weights = model.pretrained()
    
    print("Layer No: ", args.layer_no)
    print("Batch Size: ", args.batchsize)
    print("Max Data: ", max_data)
    print("-----------------------------")
    single_layer_run(args.layer_no, args.batchsize)
