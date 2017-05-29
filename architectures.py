""" 
Definitions and utilities for the flownet model

This file contains functions to define net architectures for flownet in tensorflow 

"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

def flownet_s(imgs_0, imgs_1, flows):
	"""Build the flownet_s model 
	   Check train.prototxt for original caffe model
	"""
	net = tf.concat([imgs_0, imgs_1], FLAGS.img_shape[-1],  name='concat_0')

	# stack of convolutions
	# add mean ?
	convs = {"conv1" : [64, [7,7], 2],
			"conv2_1" : [128, [5,5], 2], # _1 to concat easily later
			"conv3" : [256, [5,5], 2], 
			"conv3_1" : [256, [3,3], 1], 
			"conv4" : [512, [3,3], 2], 
			"conv4_1" : [512, [3,3], 1], 
			"conv5" : [512, [3,3], 2], 
			"conv5_1" : [512, [3,3], 1],
			"conv6" : [1024, [3,3], 2], 
			"conv6_1" : [1024, [3,3], 1], 
			}

	# from train.prototxt
	loss_weights = [0.32, 0.08, 0.02, 0.01, 0.005]

 	with slim.arg_scope([slim.conv2d],
					 	activation_fn=tf.nn.relu,
					 	normalizer_fn=slim.batch_norm,
					 	padding='SAME',
					 	weights_regularizer=slim.l2_regularizer(1e-4)):
		#convolutions + relu
		for key in sorted(convs): 
			net = slim.conv2d(net, convs[key][0], convs[key][1], convs[key][2], scope=key)

	for i in range(4):
		# flow predict 
		# no relu?
		flow_predict = slim.conv2d(net, 2, [3, 3], 1, scope='predict_flow_' + str(6-i))
		# add L1 loss
		
		[batchsize, height, width, channels] = flow_predict.get_shape().as_list()
		# resize  with ResizeMethod.BILINEAR ?
		downsample = tf.image.resize_images(flows, [height, width])
		tf.losses.absolute_difference(flow_predict, 
										downsample, 
										loss_weights[i], 
										scope='absolute_loss_'+ str(6-i))
		# Upconv flow
		# no Relu?
		flow_up = slim.conv2d_transpose(flow_predict, 2, [4, 4], 2, scope='flow_up_'+ str(6-i)+ "_to_" + str(5-i))
		# deconv + relu
		with slim.arg_scope([slim.conv2d_transpose],
					 	activation_fn=tf.nn.relu,
					 	normalizer_fn=slim.batch_norm,
					 	padding='SAME',
					 	weights_regularizer=slim.l2_regularizer(1e-4)):
			deconv = slim.conv2d_transpose(net, 512/2**i , [4, 4], 2, scope='deconv_'+ str(5-i))

		# get old convolution
		to_concat = tf.get_default_graph().get_tensor_by_name('conv'+str(5-i)+"_1/Relu:0")
		# concat convX_1, deconv, flow_up
		net = tf.concat([to_concat, deconv, flow_up], FLAGS.img_shape[-1], name='concat_' + str(5-i))

	flow_predict = slim.conv2d(net, 2, [3, 3], 1, scope='flow_pred')
	# resize  with ResizeMethod.BILINEAR as default
	flow_up = tf.image.resize_images(flow_predict, FLAGS.img_shape[:2])
	tf.losses.absolute_difference(flow_up, flows, loss_weights[4], scope='absolute_loss_'+ str(6-4))

	return flow_up

