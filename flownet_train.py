# FlowNet in Tensorflow
# Training
# ==============================================================================

import sys
import argparse
import time
import os
from os.path import dirname

import psutil
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import flownet

dir_path = dirname(os.path.realpath(__file__))

# Basic model parameters as external flags.
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                           """Initial learning rate.""")

def image_summary(imgs_0, imgs_1, text, shape, flows=None):
	if FLAGS.img_summary:
		tf.summary.image(text + "_img_0", imgs_0, 2)
		tf.summary.image(text + "_img_1", imgs_1, 2)
		if flows != None:
			hsv_flows = flownet.flows_to_hsv(flows, shape)
			tf.summary.image(text + "_flow", hsv_flows, 2)

def placeholder_inputs(img_shape, flow_shape):
	"""Generate placeholder variables to represent the input tensors."""

	batch_size = FLAGS.batch_size
	images_0_placeholder = tf.placeholder(tf.float32, shape=([batch_size] + img_shape))
	images_1_placeholder = tf.placeholder(tf.float32, shape=([batch_size] + img_shape))
	flows_placeholder = tf.placeholder(tf.float32, shape=([batch_size] + flow_shape))
	return images_0_placeholder, images_1_placeholder, flows_placeholder

def fill_feed_dict(data_lists, images_0_pl, images_1_pl, flows_pl):
	"""Fills the feed_dict for training the given step."""

	size = FLAGS.batch_size
	images, flows  = data_lists.next_batch(size)
	feed_dict = {
	  images_0_pl: images[0],
	  images_1_pl: images[1],
	  flows_pl: flows,
	}
  	return feed_dict

def run_training(last_ckpt, start_step):
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
		imgs_0_pl, imgs_1_pl, flows_pl = placeholder_inputs(img_shape, flow_shape)

		# Image / Flow Summary
		#image_summary(imgs_0_pl, imgs_1_pl, "A_", flow_shape, flows_pl)

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
		
		#augI_0_pl_2, augI_1_pl_2, augF_pl_2 = flownet.affine_trafo_2(chroI_0, chroI_1, flows_pl, flow_shape) 
		# Image / Flow Summary
		#image_summary(augI_0_pl_2, augI_1_pl_2, "C_affine2", flow_shape, augF_pl_2)

		print("--- Create Image Rotation / Scaling ---")
		#rotation / scaling (Cropping) 
		#rotI_0_pl, rotI_1_pl, rotF_pl = flownet.rotation(augI_0_pl, augI_1_pl, augF_pl, img_shape) 

		# Image / Flow Summary
		#image_summary(rotI_0_pl, rotI_1_pl, "D_rotation", flow_shape, rotF_pl)

		# initialize glabal step
		global_step = tf.Variable(start_step, name='global_step', trainable=False)

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
		saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

		# Create a session for running Ops on the Graph.
		#config=tf.ConfigProto(log_device_placement=True)
		sess = tf.Session()

		# restore model or initialize new
		if last_ckpt:
			tf.train.Saver().restore(sess, last_ckpt)
		else:
			# Add the variable initializer Op.
			init = tf.global_variables_initializer()#
			# Run the Op to initialize the variables.
			sess.run(init)

		# Instantiate a SummaryWriter to output summaries and the Graph.
		summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
		summary = tf.summary.merge_all()
		# Run options		
		run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		run_metadata = tf.RunMetadata()

		# Start the training loop.
		print("--- Start the training loop ---")
		#start time
		start = time.time()
		start_incl_eval = time.time()
		for step in range(start_step, FLAGS.steps):
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
			if step % 100 == 0:
				# Print status to stdout.
				batch_duration = time.time() - start
				print('Step %d / %d: Training loss = %.2f (%.3f sec)' % 
					(step, FLAGS.steps, loss_value, batch_duration))
				# Update the events file.
				# Build the summary Tensor based on the TF collection of Summaries.
				summary_str = sess.run(summary, feed_dict=feed_dict)
				summary_writer.add_summary(summary_str, step)
				summary_writer.flush()
				# start time again
				start = time.time()

			# Save a checkpoint and evaluate the model periodically.
			if step % 1000 == 0 or (step + 1) == FLAGS.steps:
				checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
				saver.save(sess, checkpoint_file, global_step=step)
				batch_duration = time.time() - start_incl_eval
				m, s = divmod((batch_duration)*(FLAGS.steps-step)/1000, 60)
				h, m = divmod(m, 60)
				print("Estimated termination in: %d:%02d:%02d" % (h, m, s))
				start_incl_eval = time.time()
				# print timeline for performance analysis
				tl = timeline.Timeline(run_metadata.step_stats)
				ctf = tl.generate_chrome_trace_format()
				with open('tiptop_clean/timeline/'+ str(step) +'-timeline.json', 'w') as f:
					f.write(ctf)

def main(_):
	# check where to start training / if checkpoints exists
	if not FLAGS.restore:
		print("--- Flag Retore True -> New Training --- ")
		if tf.gfile.Exists(FLAGS.log_dir):
			tf.gfile.DeleteRecursively(FLAGS.log_dir)
		tf.gfile.MakeDirs(FLAGS.log_dir)
		last_ckpt = None
		run_training(last_ckpt, 0)
	# get last ckpt if exist
	else:
		ckpt_list = sorted([f for f in os.listdir(FLAGS.log_dir) if 'model.ckpt' in f])
		if not ckpt_list:
			print("--- Empty Log -> New Training --- ")
			run_training(ckpt_list, 0)
		else:
			last_ckpt_n =  int(ckpt_list[-1].split('.')[1].split('-')[1])
			last_ckpt = FLAGS.log_dir + '/model.ckpt-'+str(last_ckpt_n)
			print("--- Loading Checkpoint --- ", last_ckpt)
			run_training(last_ckpt, last_ckpt_n)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
	  '--data_dir',
	  type=str,
	  default='data/FlyingChairs_release/data/',
	  help='Directory to put the input data.'
	)
	parser.add_argument(
	  '--split_list',
	  type=str,
	  default='data/FlyingChairs_release_test_train_split.list',
	  help='List where to split train / test'
	)
	parser.add_argument(
	  '--log_dir',
	  type=str,
	  default='tiptop_clean/log',
	  help='Directory where to write event logs and checkpoints'
	)
	parser.add_argument(
	  '--batch_size',
	  type=int,
	  default=8,
	  help='Batch Size'
	)
	parser.add_argument(
	  '--steps',
	  type=int,
	  default=500000,
	  help='Iteration Steps'
	)
	parser.add_argument(
	  '--restore',
	  type=bool,
	  default=True,
	  help='Restore from log'
	)
	parser.add_argument(
	  '--img_summary',
	  type=bool,
	  default=False,
	  help='Make image summary'
	)
	FLAGS.data_dir = os.path.join(dir_path,  parser.parse_args().data_dir)
	FLAGS.log_dir = os.path.join(dir_path, parser.parse_args().log_dir)
	FLAGS.batch_size = parser.parse_args().batch_size
	FLAGS.steps = parser.parse_args().steps
	FLAGS.restore = parser.parse_args().restore
	FLAGS.img_summary = parser.parse_args().img_summary
	FLAGS.split_list = os.path.join(dir_path,  parser.parse_args().split_list)
	tf.app.run()
