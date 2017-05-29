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

def main(_):
	"""Evaluate FlowNet for test set"""

	with tf.Graph().as_default():
		# Generate tensors from numpy images and flows.
		imgs_0, imgs_1, flows = flownet_tools.get_data(FLAGS.datadir)

		# img summary after loading
		flownet.image_summary(imgs_0, imgs_1, "A_input", flows)	

		# Get flow tensor from flownet model
		calc_flows = architectures.flownet_s(imgs_0, imgs_1, flowss)
		loss = tf.losses.get_total_loss()#
		tf.summary.scalar('Training Loss', loss)
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
		    eval_op=loss,
		    eval_interval_secs=FLAGS.eval_interval_secs,
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
	  default='log',
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
