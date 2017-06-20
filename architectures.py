"""
Definitions and utilities for the FlowNet model
This file contains functions to define net architectures for FlowNet in Tensorflow
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import flags

import numpy as np

FLAGS = flags.FLAGS
import flownet


def flownet_s(imgs_0, imgs_1, flows):
    """Build the flownet_s model
       Check train.prototxt for original caffe model
    """
    #TODO: test to normalize [0, 255] --> [-1, 1]
    #imgs_0 = tf.subtract(tf.div(imgs_0, 255/2.0), 1)
    #imgs_1 = tf.subtract(tf.div(imgs_1, 255/2.0), 1)

    net = tf.concat([imgs_0, imgs_1], -1 ,name='concat_0')
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
    loss_weights = np.array([0.32, 0.08, 0.02, 0.01, 0.005])
    #convolutions + relu
    with slim.arg_scope([slim.conv2d],
			activation_fn=tf.nn.relu,
			padding='SAME',
			#TODO: test [1e-3, 1e-4, 1e-5]
			weights_regularizer=slim.l2_regularizer(1e-4)):
    	for key, value in sorted(convs.iteritems()):
            net = slim.conv2d(net, value[0], value[1], value[2], scope=key)
    
    for i in range(4):
        # flow predict
        # no relu
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], 
			    activation_fn=None,
		            padding='SAME',
			     #TODO: test [1e-3, 1e-4, 1e-5],
                            weights_regularizer=slim.l2_regularizer(1e-4)):
            flow_predict = slim.conv2d(
                net, 2, [3, 3], 1, scope='predict_flow_' + str(6 - i))
            flow_up = slim.conv2d_transpose(
                flow_predict, 2, [4, 4], 2, scope='flow_up_' + str(6 - i) + "_to_" + str(5 - i))
        
	_, height, width, _ = flow_predict.get_shape().as_list()
        # resize  with ResizeMethod.BILINEAR ?
	#TODO: check if this sampling is correct by visualisation 
        downsample = tf.image.resize_images(flows, [height, width])
        # add L1 loss
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
    with slim.arg_scope([slim.conv2d],
                        activation_fn=None,
			padding='SAME',
                        #TODO: test [1e-3, 1e-4, 1e-5]    
			 weights_regularizer=slim.l2_regularizer(1e-4)):
	flow_predict = slim.conv2d(net, 2, [3, 3], 1, scope='flow_pred')
    # resize  with ResizeMethod.BILINEAR ?
    # check if this sampling is correct by visualisation 
    flow_up = tf.image.resize_images(flow_predict, FLAGS.img_shape[:2])
    tf.losses.absolute_difference(flow_up, flows, loss_weights[4], scope='absolute_loss_' + str(6 - 4))

    return flow_up
