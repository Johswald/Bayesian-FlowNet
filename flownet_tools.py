"""
Definitions and utilities for the FlowNet model

This file contains functions to read .jpg and .flo to tensorflow for Flownet training
and evalutation on the Flying Chairs Datatset / Singel / Kitti / et. al
"""

import glob
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import os

import flownet

FLAGS = flags.FLAGS

def tensorflow_reader(list_0, list_1, flow_list, shuffle_all, batchs):
	"""Construct tensorflow_reader for input lists.
	....
	"""
	assert len(list_0) == len(list_1) == len(flow_list) != 0, ('Input Lengths not correct')

	print("Number of inputs: " + str(len(list_0)))
	if shuffle_all == True:
		p = np.random.permutation(len(list_0))
	else:
		p = np.arange(len(list_0))
	list_0 = [list_0[i] for i in p]
	list_1 = [list_1[i] for i in p]
	flow_list = [flow_list[i] for i in p]
	input_queue = tf.train.slice_input_producer(
									[list_0, list_1],
									shuffle=False) # shuffled before
	# image reader
	content_0 = tf.read_file(input_queue[0])
	content_1 = tf.read_file(input_queue[1])
	imgs_0 = tf.image.decode_image(content_0, channels=3)
	imgs_1 = tf.image.decode_image(content_1, channels=3)
	# convert to [0, 1] images
	imgs_0 = tf.image.convert_image_dtype(imgs_0, dtype=tf.float32)
	imgs_1 = tf.image.convert_image_dtype(imgs_1, dtype=tf.float32)
	# flow reader
	filename_queue = tf.train.string_input_producer(flow_list, shuffle=False)
	record_bytes = FLAGS.record_bytes #1572876
	reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
	key, value = reader.read(filename_queue)
	record_bytes = tf.decode_raw(value, tf.float32)

	magic = tf.slice(record_bytes, [0], [1]) # .flo number 202021.25
	size  = tf.slice(record_bytes, [1], [2]) # size of flow / image
	flows = tf.slice(record_bytes, [3], [np.prod(FLAGS.flow_shape)])
	flows = tf.reshape(flows, FLAGS.flow_shape)


	# set shape
	imgs_0.set_shape(FLAGS.img_shape)
	imgs_1.set_shape(FLAGS.img_shape)
	flows.set_shape(FLAGS.flow_shape)

	return tf.train.batch([imgs_0, imgs_1, flows],
	                      batch_size=batchs
	                      #,num_threads=1
	                      )

def get_data(datadir, shuffle_all):
	"""Construct input data for FlowNetS Training.
	....
	"""
	with tf.name_scope('Input'):
		ms = FLAGS.max_steps
		list_0 = sorted(glob.glob(datadir + '*img1.jpg'))
		list_1 = sorted(glob.glob(datadir + '*img2.jpg'))
		flow_list = sorted(glob.glob(datadir + '*.flo'))
		num = int(np.ceil(ms/len(list_0)) +1)
		list_0 = list_0*num
		list_1 = list_1*num
		flow_list = flow_list*num
		return tensorflow_reader(list_0[:ms], list_1[:ms], flow_list[:ms], shuffle_all, FLAGS.batchsize)

def get_data_flow_s(datadir, shuffle_all, batchs):
	"""Construct input data.
	....
	"""
	with tf.name_scope('Input'):
		list_0 = sorted(glob.glob(datadir + '*img1.jpg'))
		list_1 = sorted(glob.glob(datadir + '*img2.jpg'))
		flow_list = sorted(glob.glob(datadir + '*.flo'))
		return tensorflow_reader(list_0, list_1, flow_list, shuffle_all, batchs)

def get_data_sintel(datadir, shuffle_all, batchs):
	"""Construct input data.
	....
	"""
	sintel_imgs = "clean/"
	sintel_flows = "flow/"
	with tf.name_scope('Input'):
		list_0 = []
		list_1 = []
		flow_list = []
		for subdir, dirs, files in os.walk(datadir + sintel_imgs):
			for c_dir in sorted(dirs):
				list_0 += sorted(glob.glob(datadir + sintel_imgs + c_dir + '/*.png'))[:-1]
				list_1 += sorted(glob.glob(datadir + sintel_imgs + c_dir + '/*.png'))[1:]
		for subdir, dirs, files in os.walk(datadir + sintel_flows):
			for c_dir in sorted(dirs):
				flow_list += sorted(glob.glob(datadir + sintel_flows + c_dir + '/*.flo'))
		return tensorflow_reader(list_0, list_1, flow_list, shuffle_all, batchs)

def get_data_middeburry(datadir, shuffle_all, batchs):
	"""Construct input data for middleburry dataset.
	....
	"""
	# TODO (oswald) NOT working yet since middeburry inputs are not the same size , resize?
	sintel_imgs = "imgs/"
	sintel_flows = "flows/"
	with tf.name_scope('Input'):
		list_0 = []
		list_1 = []
		flow_list = []
		for subdir, dirs, files in os.walk(datadir + sintel_imgs):
			for c_dir in sorted(dirs):
				list_0 += sorted(glob.glob(datadir + sintel_imgs + c_dir + '/*10.png'))
				list_1 += sorted(glob.glob(datadir + sintel_imgs + c_dir + '/*11.png'))
		for subdir, dirs, files in os.walk(datadir + sintel_flows):
			for c_dir in sorted(dirs):
				flow_list += sorted(glob.glob(datadir + sintel_flows + c_dir + '/*.flo'))
		return tensorflow_reader(list_0, list_1, flow_list, shuffle_all, batchs)


def get_data_kitti(datadir, shuffle_all, batchs):
	"""Construct input data.
	....
	"""
	sintel_imgs_1 = "image_2_crop/"
	sintel_flows = "flow_occ_crop/"
	with tf.name_scope('Input'):
		# after number 154 image sizes change
		list_0 = sorted(glob.glob(datadir + sintel_imgs_1 + '/*10.png'))
		list_1 = sorted(glob.glob(datadir + sintel_imgs_1 + '/*11.png'))
		flow_list = sorted(glob.glob(datadir + sintel_flows + '/*.png'))
		print(len(list_0), len(list_1), len(flow_list))
		print("Number of input length: " + str(len(list_0)))
		assert len(list_0) == len(list_1) == len(flow_list) != 0, ('Input Lengths not correct')

		if shuffle_all:
			p = np.random.permutation(len(list_0))
		else:
			p = np.arange(len(list_0))
		list_0 = [list_0[i] for i in p]
		list_1 = [list_1[i] for i in p]
		flow_list = [flow_list[i] for i in p]

		input_queue = tf.train.slice_input_producer(
										[list_0, list_1, flow_list],
										shuffle=False) # shuffled before
		# image reader
		content_0 = tf.read_file(input_queue[0])
		content_1 = tf.read_file(input_queue[1])
		content_flow = tf.read_file(input_queue[2])

		imgs_0 = tf.image.decode_png(content_0, channels=3)
		imgs_1 = tf.image.decode_png(content_1, channels=3)
		imgs_0 = tf.image.convert_image_dtype(imgs_0, dtype=tf.float32)
		imgs_1 = tf.image.convert_image_dtype(imgs_1, dtype=tf.float32)
		flows = tf.cast(tf.image.decode_png(content_flow, channels=3, dtype=tf.uint16), tf.float32)
		# set shape

		imgs_0.set_shape(FLAGS.img_shape)
		imgs_1.set_shape(FLAGS.img_shape)
		flows.set_shape(FLAGS.img_shape)

		return tf.train.batch([imgs_0, imgs_1, flows],
		                      batch_size=batchs
		                      #,num_threads=1
		                      )
