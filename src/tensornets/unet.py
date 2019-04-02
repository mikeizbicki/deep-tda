from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

from tensorflow.contrib.layers import convolution2d_transpose as tconv2d
from tensornets.layers import conv2d

from tensornets.ops import *
from tensornets.utils import set_args
from tensornets.utils import var_scope


def __args__(is_training):
    return [([conv2d], {'padding': 'SAME', 'activation_fn': None}),
            ([conv2d_trans], {'padding': 'SAME', 'activation_fn': None})]


@var_scope('unet')
@set_args(__args__)
def unet(x, kernel_size=5, blocks_list=[4, 5, 6, 6, 7, 7],
         is_training=False, scope=None, reuse=None):
    encodeds = []
    for (i, blocks) in enumerate(blocks_list):
        # kernel_initializer='he_normal'
        x = conv2d(x, 2 ** blocks, kernel_size, stride=2, scope="%d" % i)
        x = leaky_relu(x, name="%d/lrelu" % i)
        encodeds.append(x)

    for (i, blocks) in enumerate(blocks_list[::-1][1:]):
        x = tconv2d(x, 2 ** blocks, kernel_size, stride=2, scope="t%d" % i)
        x = relu(x, name="t%d/relu" % i)
        x = concat([x, encodeds[-2-i]], axis=3, name="t%d/concat" % i)

    x = tconv2d(x, 1, kernel_size, stride=2, scope='logits')
    x = sigmoid(x, name='probs')
    return x
