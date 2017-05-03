# FlowNet in Tensorflow
# Training
# ==============================================================================

import sys
import argparse
import time
import os
from os.path import dirname
import re
import cv2
import psutil
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline
from tensorflow.python.training import saver as tf_saver

import flownet

dir_path = dirname(os.path.realpath(__file__))

# Basic model parameters as external flags.
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                           """Initial learning rate.""")

def image_summary(imgs_0, imgs_1, text, shape, flows=None):
	if FLAGS.imgsummary:
		tf.summary.image(text + "_img_0", imgs_0, 2)
		tf.summary.image(text + "_img_1", imgs_1, 2)
		if flows != None:
			flow_imgs = flownet.flows_to_img(flows, shape)
			tf.summary.image(text + "_flow", flow_imgs, 8)

def placeholder_inputs(img_shape, flow_shape):
	"""Generate placeholder variables to represent the input tensors."""

	batchsize = FLAGS.batchsize
	images_0_placeholder = tf.placeholder(tf.float32, shape=([batchsize] + img_shape))
	images_1_placeholder = tf.placeholder(tf.float32, shape=([batchsize] + img_shape))
	flows_placeholder = tf.placeholder(tf.float32, shape=([batchsize] + flow_shape))
	return images_0_placeholder, images_1_placeholder, flows_placeholder

def fill_feed_dict(data_lists, images_0_pl, images_1_pl, flows_pl):
	"""Fills the feed_dict for training the given step."""

	size = FLAGS.batchsize
	images, flows  = data_lists.next_batch(size)
	feed_dict = {
	  images_0_pl: images[0],
	  images_1_pl: images[1],
	  flows_pl: flows,
	}
  	return feed_dict

def main(_):
	"""Train FlowNet for a number of steps."""

	# Get the lists of two images and the .flo file with a batch reader
	print("--- Start FlowNet Training ---")
	print("--- Create data list for input batch reading ---")
	data_lists = flownet.read_data_lists()
	# we (have) split the Flying Chairs dataset into 22, 232 training and 640 test samples 
	train_set = data_lists.train
	test_set = data_lists.test

	flow_shape = train_set.flow_shape
	img_shape = train_set.image_shape
	# Add the variable initializer Op.
	with tf.Graph().as_default():
		
		print("--- Create Placeholders ---")

		# Generate placeholders for the images and labels.
		images, flows_pl  = train_set.next_batch(8)
		flows_pl = tf.stack([tf.convert_to_tensor(flows_pl[i]) for i in range(8)])
		imgs_0_pl = tf.stack([tf.convert_to_tensor(images[0][i]) for i in range(8)])
		imgs_1_pl = tf.stack([tf.convert_to_tensor(images[1][i]) for i in range(8)])
		# Image / Flow Summary
		#image_summary(imgs_0_pl, imgs_1_pl, "A", flow_shape, flows_pl)

		print("--- Create chromatic Augmentation ---")
		# chromatic tranformation in images
		#chroI_0, chroI_1 = flownet.chromatic_augm(imgs_0_pl, imgs_1_pl)
		
		# Image / Flow Summary
		#image_summary(chroI_0, chroI_1,"B_chromatic",flow_shape)

		print("--- Create Affine Transformation ---")
		#affine tranformation in tf.py_func fo images and flows_pl
		#aug_data = [chroI_0, chroI_1, flows_pl]
		#augI_0_pl, augI_1_pl, augF_pl = flownet.affine_trafo(aug_data, img_shape, flow_shape) 

		# Image / Flow Summary
		#image_summary(augI_0_pl, augI_1_pl, "C_affine", flow_shape, augF_pl)

		print("--- Create Image Rotation / Scaling ---")
		#rotation / scaling (Cropping) 
		#rotI_0_pl, rotI_1_pl, rotF_pl = flownet.rotation(augI_0_pl, augI_1_pl, augF_pl, img_shape) 

		# Image / Flow Summary
		#image_summary(rotI_0_pl, rotI_1_pl, "D_rotation", flow_shape, rotF_pl)

		# initialize glabal step
		global_step = tf.Variable(0, name='global_step', trainable=False)

		print("--- Load Inference Model and add loss & train ops ---")
		# Build a Graph that computes predictions from the inference model.
		calc_flows = flownet.inference(imgs_0_pl, imgs_1_pl, img_shape)

		# Image / Flow Summary
		#image_summary(rotI_0_pl, rotI_1_pl, "E_result", flow_shape, calc_flows)

		# Add to the Graph the Ops for loss calculation.
		loss = flownet.loss(calc_flows, flows_pl, flow_shape)

		# Add to the Graph the Ops that calculate and apply gradients.
		train_op = flownet.training(loss, global_step, FLAGS.learning_rate)

		# Create a saver for writing training checkpoints.
		saver = tf_saver.Saver(max_to_keep=5,
						keep_checkpoint_every_n_hours=1)

		# Create a session for running Ops on the Graph.
		#config=tf.ConfigProto(log_device_placement=True)
		sess = tf.Session()

		# restore model or initialize new
		# Add the variable initializer Op.
		#init = tf.global_variables_initializer()#
		# Run the Op to initialize the variables.
		#sess.run(init)

		# Instantiate a SummaryWriter to output summaries and the Graph.
		summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
		summary = tf.summary.merge_all()
		# Run options		
		run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		run_metadata = tf.RunMetadata()

		# Start the training loop.
		print("--- Start the training loop ---")
		#start time
		config = tf.ConfigProto()
		#config.gpu_options.allow_growth = True
		# config.log_device_placement = True

		slim.learning.train(
			train_op,
			logdir=FLAGS.logdir + '/train',
			save_summaries_secs=150,
			save_interval_secs=300,
			master="",
			is_chief=(0 == 0),
			startup_delay_steps=(0 * 20),
			log_every_n_steps=100,
			session_config=config,
			trace_every_n_steps=1000,
			saver=saver,
			number_of_steps=FLAGS.max_steps,
		)

		start = time.time()
		start_incl_eval = time.time()
		for step in range(start_step, FLAGS.max_steps):
			"""Since, 
			in a sense, every pixel is a training sample, we use fairly small minibatches of 8 image pairs. 
			We start with learning rate lambda = 1e-4 and then divide it by 2 every 100k iterations 
			after the first 300k"""
			if step %100000 == 0:
				if step >= 300000:
					FLAGS.learning_rate = FLAGS.learning_rate/2

			# Fill a feed dictionary with the actual set of images and labels
			# for this particular training step.
			feed_dict = fill_feed_dict(train_set,
			                         imgs_0_pl, imgs_1_pl, flows_pl)

			# Run one step of the model.
			_, loss_value = sess.run([train_op, loss],
			                       feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
			
			#Write the summaries and print an overview fairly often.
			if step % 1  == 0:
				# Print status to stdout.
				batch_duration = time.time() - start
				print('Step %d / %d: Training loss = %.2f (%.3f sec)' % 
					(step, FLAGS.max_steps, loss_value, batch_duration))
				# Update the events file.
				# Build the summary Tensor based on the TF collection of Summaries.
				summary_str = sess.run(summary, feed_dict=feed_dict)
				summary_writer.add_summary(summary_str, step)
				summary_writer.flush()
				# start time again
				start = time.time()

			# Save a checkpoint and evaluate the model periodically.
			if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
				checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
				saver.save(sess, checkpoint_file, global_step=step)
				batch_duration = time.time() - start_incl_eval
				m, s = divmod((batch_duration)*(FLAGS.max_steps-step)/1000, 60)
				h, m = divmod(m, 60)
				print("Estimated termination in: %d:%02d:%02d" % (h, m, s))
				start_incl_eval = time.time()
				# print timeline for performance analysis
				tl = timeline.Timeline(run_metadata.step_stats)
				ctf = tl.generate_chrome_trace_format()
				with open('timeline/'+ str(step) +'-timeline.json', 'w') as f:
					f.write(ctf)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
	  '--datadir',
	  type=str,
	  default='data/FlyingChairs_release/data/',
	  help='Directory to put the input data.'
	)
	parser.add_argument(
	  '--splitlist',
	  type=str,
	  default='data/FlyingChairs_release_test_train_split.list',
	  help='List where to split train / test'
	)
	parser.add_argument(
	  '--logdir',
	  type=str,
	  default='tiptop_clean/log',
	  help='Directory where to write event logs and checkpoints'
	)
	parser.add_argument(
	  '--batchsize',
	  type=int,
	  default=8,
	  help='Batch Size'
	)
	parser.add_argument(
	  '--max_steps',
	  type=int,
	  default=600000,
	  help='Iteration Steps'
	)
	parser.add_argument(
	  '--restore',
	  type=bool,
	  default=True,
	  help='Restore from log'
	)
	parser.add_argument(
	  '--imgsummary',
	  type=bool,
	  default=False,
	  help='Make image summary'
	)
	FLAGS.datadir = os.path.join(dir_path,  parser.parse_args().datadir)
	FLAGS.logdir = os.path.join(dir_path, parser.parse_args().logdir)
	FLAGS.batchsize = parser.parse_args().batchsize
	FLAGS.max_steps = parser.parse_args().max_steps
	FLAGS.restore = parser.parse_args().restore
	FLAGS.imgsummary = parser.parse_args().imgsummary
	FLAGS.splitlist = os.path.join(dir_path,  parser.parse_args().splitlist)
	tf.app.run()
