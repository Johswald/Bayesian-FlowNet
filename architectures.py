# FlowNet in Tensorflow
# ==============================================================================

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

def flownet_s(imgs_0, imgs_1):
	"""Build the flownet model """
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

 	with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
					  activation_fn=tf.nn.relu,
					  weights_regularizer=slim.l2_regularizer(1e-4)):
		#convolutions
		for key in sorted(convs): 
			net = slim.conv2d(net, convs[key][0], convs[key][1], convs[key][2], scope=key)
		#print([n.name for n in tf.get_default_graph().as_graph_def().node])
		# deconv + flow
		for i in range(4):
			# flow predict + flow deconv
			flow_predict = slim.conv2d(net, 2, [3, 3], 1, scope='flow_' + str(6-i))
			flow_up = slim.conv2d_transpose(flow_predict, 2, [4, 4], 2, scope='flow_dv_'+ str(6-i))
			# devonv net + concat
			deconv = slim.conv2d_transpose(net, 512/2**i , [4, 4], 2, scope='deconv_'+ str(5-i))
			# get old convolution
			to_concat = tf.get_default_graph().get_tensor_by_name('conv'+str(5-i)+"_1/Relu:0")
			net = tf.concat([deconv, to_concat, flow_up], FLAGS.img_shape[-1], name='concat_' + str(i+1))
		flow_predict = slim.conv2d(net, 2, [3, 3], 1, scope='flow_pred')
	# resize  with ResizeMethod.BILINEAR as default
	flow_up = tf.image.resize_images(flow_predict, FLAGS.img_shape[:2])
	return flow_up


def flownet_2(imgs_0, imgs_1):
	"""Build the flownet model """
	net = tf.concat([imgs_0, imgs_1], FLAGS.img_shape[-1],  name='concat_0')
	# stack of convolutions
	# add mean ?
	convs = {"conv1" : [64, [7,7], 2],
			"conv1_1" : [64, [7,7], 1],
			"conv2" : [128, [5,5], 2], 
			"conv2_1" : [128, [5,5], 1], 
			"conv3" : [256, [5,5], 2], 
			"conv3_1" : [256, [3,3], 1], 
			"conv4" : [512, [3,3], 2], 
			"conv4_1" : [512, [3,3], 1], 
			"conv5" : [512, [3,3], 2], 
			"conv5_1" : [512, [3,3], 1],
			"conv6" : [1024, [3,3], 2], 
			"conv6_1" : [1024, [3,3], 1], 
			}

 	with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
					  activation_fn=tf.nn.relu,
					  weights_regularizer=slim.l2_regularizer(1e-4)):
		#convolutions
		for key in sorted(convs): 
			net = slim.conv2d(net, convs[key][0], convs[key][1], convs[key][2], scope=key)
		#print([n.name for n in tf.get_default_graph().as_graph_def().node])
		# deconv + flow
		for i in range(5):
			# flow predict + flow deconv
			flow_predict = slim.conv2d(net, 2, [3, 3], 1, scope='flow_' + str(6-i))
			flow_up = slim.conv2d_transpose(flow_predict, 2, [4, 4], 2, scope='flow_dv_'+ str(6-i))
			# devonv net + concat
			deconv = slim.conv2d_transpose(net, 512/2**i , [4, 4], 2, scope='deconv_'+ str(5-i))
			# get old convolution
			to_concat = tf.get_default_graph().get_tensor_by_name('conv'+str(5-i)+"_1/Relu:0")
			net = tf.concat([deconv, to_concat, flow_up], FLAGS.img_shape[-1], name='concat_' + str(i+1))
		flow_predict = slim.conv2d(net, 2, [3, 3], 1, scope='flow_pred')
	# resize  with ResizeMethod.BILINEAR as default
	flow_up = tf.image.resize_images(flow_predict, FLAGS.img_shape[:2])
	return flow_up

def flownet_noresize(imgs_0, imgs_1):
	"""Build the flownet model """
	net = tf.concat([imgs_0, imgs_1], FLAGS.img_shape[-1],  name='concat_0')
	# stack of convolutions
	# add mean ?
	convs = {"conv1" : [64, [7,7], 2],
			"conv1_1" : [64, [7,7], 1],
			"conv2" : [128, [5,5], 2], 
			"conv2_1" : [128, [5,5], 1], 
			"conv3" : [256, [5,5], 2], 
			"conv3_1" : [256, [3,3], 1], 
			"conv4" : [512, [3,3], 2], 
			"conv4_1" : [512, [3,3], 1], 
			"conv5" : [512, [3,3], 2], 
			"conv5_1" : [512, [3,3], 1],
			"conv6" : [1024, [3,3], 2], 
			"conv6_1" : [1024, [3,3], 1], 
			}

 	with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
					  activation_fn=tf.nn.relu,
					  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
					  weights_regularizer=slim.l2_regularizer(0.0005)):
		#convolutions
		for key in sorted(convs): 
			net = slim.conv2d(net, convs[key][0], convs[key][1], convs[key][2], scope=key)
		#print([n.name for n in tf.get_default_graph().as_graph_def().node])
		# deconv + flow
		for i in range(5):
			# flow predict + flow deconv
			flow_predict = slim.conv2d(net, 2, [3, 3], 1, scope='flow_' + str(6-i))
			flow_up = slim.conv2d_transpose(flow_predict, 2, [4, 4], 2, scope='flow_dv_'+ str(6-i))
			# devonv net + concat
			deconv = slim.conv2d_transpose(net, 512/2**i , [4, 4], 2, scope='deconv_'+ str(5-i))
			# get old convolution
			to_concat = tf.get_default_graph().get_tensor_by_name('conv'+str(5-i)+"_1/Relu:0")
			net = tf.concat([deconv, to_concat, flow_up], FLAGS.img_shape[-1], name='concat_' + str(i+1))
		# resize  with ResizeMethod.BILINEAR as default
		flow_predict = slim.conv2d(net, 2, [3, 3], 1, scope='flow_' + str(1))
		flow_up = slim.conv2d_transpose(flow_predict, 2, [4, 4], 2, scope='flow_dv_'+ str(1))
	return flow_up
