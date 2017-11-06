"""
Definitions and utilities for the FlowNet model
This file contains functions to define net architectures for FlowNet in Tensorflow
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

import numpy as np
import math

import flownet
from concrete_drop import ConcreteDropout

def lrelu(x, leak=0.1, name="lrelu"):
    """
    leaky relu
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """

    stddev = math.sqrt(2. / (kl**2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)

def flownet_s(imgs_0, imgs_1, flows):

    """Build the flownet_s model
       Check train.prototxt for original caffe model
    """
    img_height = tf.cast(FLAGS.d_shape_img[0], tf.float32)

    with tf.name_scope('Normalization'):
        # "normalize" to [-0.5, 0.5]
        imgs_0 -= 0.5
        imgs_1 -= 0.5

    # concat images for FlowNetS architecture
    net = tf.concat([imgs_0, imgs_1], -1,name='concat_0')
    # stack of convolutions
    convs = {"conv1": [64, [7, 7], 2],
             "conv2_1": [128, [5, 5], 2],  # _1 to concat easily later
	         "conv3": [256, [5, 5], 2],
             "conv3_1": [256, [3, 3], 1],
             "conv4": [512, [3, 3], 2],
             "conv4_1": [512, [3, 3], 1],
             "conv5": [512, [3, 3], 2],
             "conv5_1": [512, [3, 3], 1],
             "conv6": [1024, [3, 3], 2],
             "conv6_1": [1024, [3, 3], 1],
             }

    #loss weights
    loss_weights = np.array([0.32, 0.08, 0.02, 0.01, 0.005])

    # set batch normalization
    if FLAGS.batch_normalization:
        normalizer = slim.batch_norm
        is_training = FLAGS.is_training
    else:
        normalizer = None
        is_training = False

    #convolutions + relu
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], weights_regularizer=FLAGS.weights_reg):
        for key, value in sorted(convs.iteritems()):
            net = slim.conv2d(net, value[0], value[1], value[2], scope=key,
                            activation_fn=lrelu,
                            normalizer_fn=normalizer,
                            normalizer_params={'is_training': is_training, 'decay': 0.9,
                                                'epsilon': 1e-5, 'updates_collections': None},
                            weights_initializer=msra_initializer(value[1][0], value[0]))

            if FLAGS.dropout:
                if not FLAGS.is_training:
                    # weight scaling due to dropout
                    weight = tf.get_default_graph().get_tensor_by_name(key + "/weights:0")
                    weight *= 1/FLAGS.drop_rate
                elif "4_1" in key or "5_1" in key or "6_1" in key:
                    net = slim.dropout(net, 1 - FLAGS.drop_rate, scope='dropout_'+ key)


        # Number of upconvolutions
        for i in range(4):
            # flow predict
            flow_predict = slim.conv2d(
                            net, 2, [3, 3], 1, scope='predict_flow_' + str(6 - i),
                            weights_initializer=msra_initializer(3, 2),
                            activation_fn=None)

            # upconvolve flow predict
            flow_up = slim.conv2d_transpose(
                            flow_predict, 2, [4, 4], 2, scope='flow_up_' + str(6 - i) + "_to_" + str(5 - i),
                            weights_initializer=msra_initializer(4, 2),
                            activation_fn=None)

            # downsample for loss
            _, height, width, _ = flow_predict.get_shape().as_list()
            # since we downsample we must change the flow pointer as well
            # originally this downsample is done through a weighted average -> might improve results
            downsample = tf.image.resize_bilinear(flows, [height, width])*height/img_height
            # add L1 loss
            tf.losses.absolute_difference(flow_predict, downsample,
                                        loss_weights[i], scope='absolute_loss_' + str(6 - i))

            # deconvolve
            deconv = slim.conv2d_transpose(net, 512 / 2**i, [4, 4], 2, scope='deconv_' + str(5 - i),
                                           activation_fn=lrelu,
                                           normalizer_fn=normalizer,
                                           normalizer_params={'is_training': is_training, 'decay': 0.9,
                                                                'epsilon': 1e-5, 'updates_collections': None},
                                           weights_initializer=msra_initializer(4, 512 / 2**i))

            # dropout
            if FLAGS.dropout and i < 2:
                if not FLAGS.is_training:
                    # weight scaling due to dropout
                    weight = tf.get_default_graph().get_tensor_by_name('deconv_' + str(5 - i) + "/weights:0")
                    weight *= 1/FLAGS.drop_rate
                else:
                    deconv = slim.dropout(deconv, 1 - FLAGS.drop_rate, scope='dropout_' +str(5-i))

            # get old convolution

            to_concat = tf.get_default_graph().get_tensor_by_name('conv' + str(5 - i) + "_1/lrelu/add:0")
            # concat convX_1, deconv, flow_up
            net = tf.concat([to_concat, deconv, flow_up], -1, name='concat_' + str(5 - i))

        # last prediction
        flow_predict = slim.conv2d(net, 2, [3, 3], 1, scope='predict_flow_2',
                                    weights_initializer=msra_initializer(3,2),
                                    activation_fn=None)

        _, height, width, _ = flow_predict.get_shape().as_list()
        downsample = tf.image.resize_nearest_neighbor(flows, [height, width])*height/img_height
        tf.losses.absolute_difference(flow_predict, downsample, loss_weights[4], scope='absolute_loss_' + str(6 - 4))

    # scale flow to orgiginal size (also for Sintel, Kitti, ...)
    flow_up = tf.image.resize_bilinear(flow_predict, FLAGS.d_shape_img[:2])
    # scale flow pointers up as well
    flow_up *= img_height/height
    return flow_up
