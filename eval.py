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

dir_path = dirname(os.path.realpath(__file__))

# Basic model parameters as external flags.
FLAGS = flags.FLAGS

flags.DEFINE_string('splitlist', 'data/FlyingChairs_release_test_train_split.list',
                           'List where to split train / test')

flags.DEFINE_integer('batchsize', 8, 'Batch size for eval loop.')

flags.DEFINE_integer('eval_interval_secs', 300,
                     'How many seconds between executions of the eval loop.')
	
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

	# Get the lists of two images and the .flo file with a batch reader
	data_lists = flownet.read_data_lists()
	# we (have) split the Flying Chairs dataset into 22, 232 training and 640 test samples 
	test_set = data_lists.test
	print(test_set._num_examples)
	imgs_np, flows_np  = test_set.next_batch(FLAGS.batchsize)
	# Add the variable initializer Op.
	with tf.Graph().as_default():
		# Generate tensors from numpy images and flows.
		imgs_0, imgs_1, flows = flownet.convert_to_tensor(imgs_np, flows_np)
		flownet.image_summary(imgs_0, imgs_1, "Ground Trouth", flows)

		# Get flow tensor from flownet model
		calc_flows = flownet.inference(imgs_0, imgs_1)
		loss = flownet.train_loss(calc_flows, flows)
		flownet.image_summary(imgs_0, imgs_1, "Result", calc_flows)
		# Run the actual evaluation loop.
		num_batches = math.ceil(test_set._num_examples / float(FLAGS.batchsize))

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
	  default='data/FlyingChairs_release/data/',
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
