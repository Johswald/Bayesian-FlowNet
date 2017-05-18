""" 
Definitions and utilities for the flownet model

This file contains functions .jpg and .flo tensorflow reader for Flownet training and evalutation
on the Flying Chairs Datatset.

"""
import glob
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

import computeColor

FLAGS = flags.FLAGS

def get_data(datadir):
	"""Construct input data.
	....
	"""
	list_0 = sorted(glob.glob(datadir + '*img1.jpg'))
	list_1 = sorted(glob.glob(datadir + '*img2.jpg'))
	flow_list = sorted(glob.glob(datadir + '*.flo'))
	assert len(list_0) == len(list_1) == len(flow_list), ('Input Lengths not correct')
	# shuffle 
	p = np.random.permutation(len(list_0))
	list_0 = [list_0[i] for i in p]
	list_1 = [list_1[i] for i in p]
	flow_list = [flow_list[i] for i in p]

	input_queue = tf.train.slice_input_producer(
									[list_0, list_1], 
									shuffle=False) # shuffled before
	# image reader
	content_0 = tf.read_file(input_queue[0])
	content_1 = tf.read_file(input_queue[1])
	imgs_0 = tf.image.decode_jpeg(content_0, channels=3)
	imgs_1 = tf.image.decode_jpeg(content_1, channels=3)
	imgs_0 = tf.image.convert_image_dtype(imgs_0, tf.float32)
	imgs_1 = tf.image.convert_image_dtype(imgs_1, tf.float32)

	# flow reader
	filename_queue = tf.train.string_input_producer(flow_list, shuffle=False)
	record_bytes = 1572876
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

	return tf.train.batch( [imgs_0, imgs_1, flows],
	                      batch_size=FLAGS.batchsize
	                      #,num_threads=1
	                      )