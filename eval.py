# FlowNet in Tensorflow
# Evaluation
# ==============================================================================

import argparse
import os
from os.path import dirname

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline
from tensorflow.python.platform import flags
from tensorflow.python.training import saver as tf_saver
import math

import flownet
import flownet_tools
import architectures

dir_path = dirname(os.path.realpath(__file__))

# Basic model parameters as external flags.
FLAGS = flags.FLAGS

flags.DEFINE_integer('batchsize', 8, 'Batch size for eval loop.')

flags.DEFINE_integer('eval_interval_secs', 300,
                     'How many seconds between executions of the eval loop.')

flags.DEFINE_integer('testsize', 640,
                     'Number of test samples')

flags.DEFINE_integer('img_shape', [384, 512, 3],
                           'Image shape: width, height, channels')

flags.DEFINE_integer('flow_shape', [384, 512, 2],
                           'Image shape: width, height, 2')

flags.DEFINE_string('master', '',
                    'BNS name of the TensorFlow master to use.')

flags.DEFINE_integer('img_summary_num', 2,
                           'Number of images in summary')

def slice_vector(vec):
	x = tf.slice(vec,[0, 0, 0, 0], [FLAGS.batchsize] + FLAGS.img_shape[:2] + [1])
	y = tf.slice(vec,[0, 0, 0, 1], [FLAGS.batchsize] + FLAGS.img_shape[:2] + [1])
	return tf.squeeze(x), tf.squeeze(y) 

def main(_):
	"""Evaluate FlowNet for test set"""

	with tf.Graph().as_default():
		# Generate tensors from numpy images and flows.
		imgs_0, imgs_1, flows = flownet_tools.get_data(FLAGS.datadir)

		# img summary after loading
		flownet.image_summary(imgs_0, imgs_1, "A_input", flows)

		# Get flow tensor from flownet model
		calc_flows = architectures.flownet_s(imgs_0, imgs_1, flows)

		# calc EPE / AEE = ((x1-x2)^2 + (y1-y2)^2)^1/2
		# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3478865/
		square = tf.square(flows - calc_flows)
		x , y = slice_vector(square)
		sqr = tf.sqrt(tf.add(x, y))
		aee = tf.metrics.mean(sqr)

		# calc EPE / AEE = cos^-1(v1*v2) where v1,v2 normalized flow vectors
		#TODO: get aae to work
		x_f , y_f = slice_vector(flows)
        	x_cf , y_cf = slice_vector(calc_flows)

 	       	length_f = tf.norm(flows, axis= -1)
       		length_cf = tf.norm(calc_flows, axis= -1)
		length_cf = tf.where(tf.not_equal(length_cf, tf.zeros_like(length_cf)), length_cf, tf.ones_like(length_cf))
        	length_f  = tf.where(tf.not_equal(length_f, tf.zeros_like(length_f)), length_f, tf.ones_like(length_f))
		dot = tf.add(tf.multiply(x_f, x_cf), tf.multiply(y_f, y_cf))
		aae = tf.metrics.mean(tf.acos(dot))

        	metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
              	      "AEE": slim.metrics.streaming_mean(aee),
        	      #"AAE": slim.metrics.streaming_mean(aae),
	        })
		
		for name, value in metrics_to_values.iteritems():
            		tf.summary.scalar(name, value)
		# Define the summaries to write:
		flownet.image_summary(None, None, "Result", calc_flows)
		# Run the actual evaluation loop.
		num_batches = math.ceil(FLAGS.testsize / float(FLAGS.batchsize))

		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		slim.evaluation.evaluation_loop(
			master=FLAGS.master,
		    checkpoint_dir=FLAGS.logdir + '/train',
		    logdir=FLAGS.logdir + '/eval',
		    num_evals=num_batches,
		    eval_op=metrics_to_updates.values(),
		    eval_interval_secs=FLAGS.eval_interval_secs,
		    summary_op=tf.summary.merge_all(),
		    session_config=config,
		    timeout=60 * 60
        )

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
	  '--datadir',
	  type=str,
	  default='data/test/',
	  help='Directory to put the input data.'
	)
	parser.add_argument(
	  '--logdir',
	  type=str,
	  default='log_flownet_s',
	  help='Directory where to write event logs and checkpoints'
	)
	parser.add_argument(
	  '--imgsummary',
	  type=bool,
	  default=True,
	  help='Make image summary'
	)

	FLAGS.datadir = os.path.join(dir_path,  parser.parse_args().datadir)
	FLAGS.logdir = os.path.join(dir_path, parser.parse_args().logdir)
	FLAGS.imgsummary = parser.parse_args().imgsummary
	tf.app.run()
