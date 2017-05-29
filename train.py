""" 
Flownet training module

"""

import sys
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

import flownet
import flownet_tools
import architectures

dir_path = dirname(os.path.realpath(__file__))

# Basic model parameters as external flags.
FLAGS = flags.FLAGS

flags.DEFINE_integer('batchsize', 8, 'Batch size.')

flags.DEFINE_integer('img_shape', [384, 512, 3],
					'Image shape: width, height, channels')

flags.DEFINE_integer('flow_shape', [384, 512, 2],
					'Image shape: width, height, 2')

flags.DEFINE_boolean('augmentation', False,
					'Use data augmentation')

flags.DEFINE_integer('boundaries', [300000, 400000],
					'boundaries for learning rate')

flags.DEFINE_integer('values', [1e-4, (1e-4)/2, (1e-4)/2/2],
					'learning rate values')

flags.DEFINE_integer('img_summary_num', 2,
					'Number of images in summary')

flags.DEFINE_integer('max_checkpoints', 5,
					'Maximum number of recent checkpoints to keep.')

flags.DEFINE_float('keep_checkpoint_every_n_hours', 5.0,
					'How often checkpoints should be kept.')

flags.DEFINE_integer('save_summaries_secs', 60,
					'How often should summaries be saved (in seconds).')

flags.DEFINE_integer('save_interval_secs', 300,
					'How often should checkpoints be saved (in seconds).')

flags.DEFINE_integer('log_every_n_steps', 100,
					'Logging interval for slim training loop.')

flags.DEFINE_integer('trace_every_n_steps', 1000,
					'Logging interval for trace.')

flags.DEFINE_integer('max_steps', 500000, 
					'Number of training steps.')

def apply_augmentation(imgs_0, imgs_1, flows):
	# apply augmenation to data batch
	
	if FLAGS.augmentation:
		with tf.name_scope('Augmentation'):
		
			#affine tranformation in tf.py_func fo images and flows_pl
			imgs_0, imgs_1, flows = flownet.affine_augm(imgs_0, imgs_1, flows) 

			#rotation / scaling (Cropping) 
			imgs_0, imgs_1, flows = flownet.rotation_crop(imgs_0, imgs_1, flows) 

			# chromatic tranformation in imagess
			imgs_0, imgs_1 = flownet.chromatic_augm(imgs_0, imgs_1)

		flownet.image_summary(imgs_0, imgs_1, "B_after_augm", flows)
	return imgs_0, imgs_1, flows

def main(_):
	"""Train FlowNet for a FLAGS.max_steps."""

	with tf.Graph().as_default():

		imgs_0, imgs_1, flows, flow_imgs = flownet_tools.get_data(FLAGS.datadir)

		# img summary after loading
		#flownet.image_summary(imgs_0, imgs_1, "A_input", flows)

		# apply augmentation
		imgs_0, imgs_1, flows = apply_augmentation(imgs_0, imgs_1, flows)

		# model
		calc_flows = architectures.flownet_s(imgs_0, imgs_1, flows)

		# img summary of result
		flownet.image_summary(None, None, "E_result", calc_flows)

		global_step = slim.get_or_create_global_step()

		train_op = flownet.create_train_op(global_step)

		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True

		saver = tf_saver.Saver(max_to_keep=FLAGS.max_checkpoints,
						keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours)

		slim.learning.train(
			train_op,
			logdir=FLAGS.logdir + '/train',
			save_summaries_secs=FLAGS.save_summaries_secs,
			save_interval_secs=FLAGS.save_interval_secs,
			summary_op=tf.summary.merge_all(),
			log_every_n_steps=FLAGS.log_every_n_steps,
			trace_every_n_steps=FLAGS.trace_every_n_steps,
			session_config=config,
			saver=saver,
			number_of_steps=FLAGS.max_steps,
		)
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
	  '--datadir',
	  type=str,
	  default='data/train/',
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
