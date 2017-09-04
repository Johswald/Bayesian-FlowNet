"""
Definitions and utilities for the flownet model
This file contains functions to define net architectures for flownet in tensorflow
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import flags

import numpy as np

FLAGS = flags.FLAGS
import flownet


def flownet_advanced(imgs_0, imgs_1, flows):
    """Build the flownet_s model
       Check train.prototxt for original caffe model
    """
    # normalize [0, 255] --> [-1, 1]
    #imgs_0 = tf.subtract(tf.div(imgs_0, 255/2.0), 1)
    #imgs_1 = tf.subtract(tf.div(imgs_1, 255/2.0), 1)

    net = tf.concat([imgs_0, imgs_1], FLAGS.img_shape[-1],  name='concat_0')

    # VGG16
    with arg_scope([layers.conv2d, layers_lib.max_pool2d], outputs_collections=end_points_collection):
        with arg_scope([layers.conv2d, layers_lib.max_pool2d], outputs_collections=end_points_collection):
            net = layers_lib.repeat(net, 2, layers.conv2d, 64, [3, 3], scope='conv1')
            net = layers_lib.repeat(net, 1, layers.conv2d, 64, [3, 3], scope='conv1_1')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
            net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
            net = layers_lib.repeat(net, 1, layers.conv2d, 128, [3, 3], scope='conv2_1')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
            net = layers_lib.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
            net = layers_lib.repeat(net, 1, layers.conv2d, 256, [3, 3], scope='conv3_1')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
            net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
            net = layers_lib.repeat(net, 1, layers.conv2d, 512, [3, 3], scope='conv4_1')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
            net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
            net = layers_lib.repeat(net, 1, layers.conv2d, 512, [3, 3], scope='conv5_1')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')
            net = layers_lib.repeat(net, 3, layers.conv2d, 1024, [3, 3], scope='conv6')
            net = layers_lib.repeat(net, 1, layers.conv2d, 1024, [3, 3], scope='conv6_1')
          # Use conv2d instead of fully_connected layers.

    # from train.prototxt
    loss_weights = np.array([0.32, 0.08, 0.02, 0.01, 0.005])
    #convolutions + relu
    for key in sorted(convs):
        net = slim.conv2d(net, convs[key][0],
                          convs[key][1], convs[key][2], scope=key)
    for i in range(4):
        # flow predict
        # no relu
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=None):
            flow_predict = slim.conv2d(
                net, 2, [3, 3], 1, scope='predict_flow_' + str(6 - i))
            flow_up = slim.conv2d_transpose(
                flow_predict, 2, [4, 4], 2, scope='flow_up_' + str(6 - i) + "_to_" + str(5 - i))
        # add L1 loss
        [batchsize, height, width, channels] = flow_predict.get_shape().as_list()
        # resize  with ResizeMethod.BILINEAR ?
        downsample = tf.image.resize_images(flows, [height, width])
        tf.losses.absolute_difference(flow_predict,
                                      downsample,
                                      loss_weights[i],
                                      scope='absolute_loss_' + str(6 - i))

        # deconv + relu
        deconv = slim.conv2d_transpose(net, 512 / 2**i, [4, 4], 2, scope='deconv_' + str(5 - i))

        # get old convolution
        to_concat = tf.get_default_graph().get_tensor_by_name('conv' + str(5 - i) + "_1/Relu:0")
        # concat convX_1, deconv, flow_up
        net = tf.concat([to_concat, deconv, flow_up], FLAGS.img_shape[-1], name='concat_' + str(5 - i))

    # last prediction
    with slim.arg_scope([slim.conv2d], activation_fn=None):
        flow_predict = slim.conv2d(net, 2, [3, 3], 1, scope='flow_pred')
    # resize  with ResizeMethod.BILINEAR as default
    flow_up = tf.image.resize_images(flow_predict, FLAGS.img_shape[:2])
    tf.losses.absolute_difference(flow_up, flows, loss_weights[4], scope='absolute_loss_' + str(6 - 4))

    return flow_up

def img0_plus_flow_warp_error(imgs_0, flows, imgs_1):
    """Pyfunc wrapper for randomized affine transformations."""

    def _warp_error(imgs_0, flows, imgs_1):

        h, w, ch = imgs_0[0].shape
        full_error = np.array([0], np.float32)
        # attempt to warp img1 to img2 with given flow
        # difficult since pixels can be warped to same pixel in img1
        indices = np.indices((h, w))
        for img_0, flow, img_1, i in zip(imgs_0, flows, imgs_1, range(FLAGS.batchsize)):
            new = np.ones((h, w, ch), np.float32) * -1
            flow = np.rint(flow)
            print(np.amax(flow))
            print(np.amin(flow))
            # print(indices.reshape(flow.shape))
            displacement = np.zeros(flow.shape, np.int16)

            displacement[:, :, 1] = indices[0] + flow[:, :, 1]
            displacement[:, :, 0] = indices[1] + flow[:, :, 0]

            # eliminate displacements outside of image
            displacement[np.where(displacement[:, :, 0] < 0)] = [0, 0]
            displacement[np.where(displacement[:, :, 1] < 0)] = [0, 0]
            displacement[np.where(displacement[:, :, 1] >= h)] = [0, 0]
            displacement[np.where(displacement[:, :, 0] >= w)] = [0, 0]

            # asign img_0
            # this is not correct since we have a problem when two pixels flow to same destination!
            new[displacement[:, :, 1], displacement[:, :, 0]] = img_0

            # asign img_1 values where we moved stuff
            new[np.where(new < 0)] = img_1[np.where(new < 0)]
            imgs_1[i] = new
        return [imgs_1]

    img_new = tf.py_func(_warp_error, [imgs_0, flows, imgs_1],
                         [tf.float32], name='imgs_plus_flow')

    return tf.squeeze(tf.stack(img_new))


def flownet_no_gt(imgs_0, imgs_1):
    """Build the flownet model with no ground truth

    """
    # normalize [0, 255] --> [-1, 1]
    #imgs_0 = tf.subtract(tf.div(imgs_0, 255/2.0), 1)
    #imgs_1 = tf.subtract(tf.div(imgs_1, 255/2.0), 1)

    net = tf.concat([imgs_0, imgs_1], FLAGS.img_shape[-1],  name='concat_0')

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

    # from train.prototxt
    loss_weights = 1 - np.array([0.32, 0.08, 0.02, 0.01, 0.005])
    #convolutions + relu
    for key in sorted(convs):
        net = slim.conv2d(net, convs[key][0],
                          convs[key][1], convs[key][2], scope=key)

    for i in range(4):
        # flow predict
        # no relu?
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=None):
            flow_predict = slim.conv2d(
                net, 2, [3, 3], 1, scope='predict_flow_' + str(6 - i))
            flow_up = slim.conv2d_transpose(
                flow_predict, 2, [4, 4], 2, scope='flow_up_' + str(6 - i) + "_to_" + str(5 - i))

        # add L1 loss with sifted img_0

        [batchsize, height, width, channels] = flow_predict.get_shape().as_list()
        # resize  with ResizeMethod.BILINEAR ?

        downsample_0 = tf.image.resize_images(imgs_0, [height, width])
        downsample_1 = tf.image.resize_images(imgs_1, [height, width])
        shifted_imgs = img0_plus_flow_warp_error(
            downsample_0, flow_predict, downsample_1)
        shifted_imgs.set_shape([batchsize, height, width, channels + 1])

        tf.losses.absolute_difference(downsample_1,
                                      shifted_imgs,
                                      loss_weights[i],
                                      scope='absolute_loss_' + str(6 - i))
        # Upconv flow
        # no Relu?"""
        # deconv + relu

        deconv = slim.conv2d_transpose(
            net, 512 / 2**i, [4, 4], 2, scope='deconv_' + str(5 - i))

        # get old convolution
        to_concat = tf.get_default_graph().get_tensor_by_name(
            'conv' + str(5 - i) + "_1/Relu:0")
        # concat convX_1, deconv, flow_up
        net = tf.concat([to_concat, deconv, flow_up],
                        FLAGS.img_shape[-1], name='concat_' + str(5 - i))

    with slim.arg_scope([slim.conv2d], activation_fn=None):
        flow_predict = slim.conv2d(net, 2, [3, 3], 1, scope='flow_pred')
    # resize  with ResizeMethod.BILINEAR as default

    flow_up = tf.image.resize_images(flow_predict, FLAGS.img_shape[:2])

    shifted_imgs = img0_plus_flow_warp_error(imgs_0, flow_up, imgs_1)
    shifted_imgs.set_shape([FLAGS.batchsize] + FLAGS.img_shape)

    flownet.image_summary(imgs_0, shifted_imgs, "Shifted", None)
    tf.losses.absolute_difference(imgs_1,
                                  shifted_imgs,
                                  loss_weights[i],
                                  scope='absolute_loss_' + str(6 - i))
    return flow_up
