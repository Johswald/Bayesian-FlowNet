"""
Definitions and utilities for the flownet model
This file contains functions for data augmentation, summary and training ops for tensorflow training
"""

import os
import cv2
import numpy as np
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import flags
import ast

import computeColor
import bilateral_solver_var as bills
import writeFlowFile

from skimage.io import imread, imsave

FLAGS = flags.FLAGS

flags.DEFINE_float('max_rotate_angle', 0.17,
						'max rotation angle')

def affine_augm(imgs_0, imgs_1, flows):
	"""Pyfunc wrapper for randomized affine transformations."""

	def _affine_transform(imgs_0, imgs_1, flows):
		"""Affine Transformation with cv2 (warpAffine)"""

		bs = FLAGS.batchsize
		h, w, ch = FLAGS.img_shape
		c = np.float32([w, h]) / 2.0
		mat = np.random.normal(size=[bs, 2, 3])
		mat[:, :2, :2] = mat[:, :2, :2] * 0.15 + np.eye(2)
		mat[:, :, 2] = mat[:, :, 2] * 2 + c - mat[:, :2, :2].dot(c)
		for mat_i, img_0, img_1, flow, i in zip(mat, imgs_0, imgs_1, flows, range(bs)):
			aug_0 = cv2.warpAffine(img_0, mat_i, (w, h), borderMode=3)
			aug_1 = cv2.warpAffine(img_1, mat_i, (w, h), borderMode=3)
			aug_f = cv2.warpAffine(flow, mat_i, (w, h), borderMode=3)
			if np.random.rand() > 0.75:
				aug_0 = cv2.GaussianBlur(aug_0, (7, 7), -1)
				aug_1 = cv2.GaussianBlur(aug_1, (7, 7), -1)
				aug_f = cv2.GaussianBlur(aug_f, (7, 7), -1)
			imgs_0[i] = aug_0
			imgs_1[i] = aug_1
			flows[i] = aug_f
	 	return [imgs_0, imgs_1, flows]

	shape = FLAGS.img_shape
	aug_data = tf.py_func( _affine_transform, [imgs_0, imgs_1, flows],
					[tf.float32, tf.float32, tf.float32], name='affine_transform')
	augI_0, augI_1, augF = aug_data[:]
	augI_0.set_shape([FLAGS.batchsize] + list(FLAGS.img_shape))
	augI_1.set_shape([FLAGS.batchsize] + list(FLAGS.img_shape))
	augF.set_shape([FLAGS.batchsize] + list(FLAGS.flow_shape))

	# Image / Flow Summary
	#image_summary(augI_0, augI_1, "B_affine", augF)
	return augI_0, augI_1, augF

def chromatic_augm(imgs_0, imgs_1):
	"""TODO: Check chromatic data augm examples in the web"""
	"""chromatic augmentation. (brightness, contrast, gamma, and color)
	(- The Gaussian noise has a sigma uniformly sampled
	from [0, 0.04]; Gaussian Blur in Affine Trafo)
	- contrast is sampled within [-0.8, 0.4];
	- multiplicative color changes to the RGB channels per image from [0.5, 2];
	- gamma values from [0.7, 1.5] and
	- additive brightness changes using Gaussian with a sigma of 0.2.
	"""
	bs = FLAGS.batchsize
	# multiplicative color changes to the RGB channels per image from [0.5, 2];
	# 1. Own testet replacement with saturation / hue
	# 2. gamma values from [0.7, 1.5] and
	# 3. Own testet brightness changes
	# different transformation in batch
	hue = np.random.uniform(-1, 1, bs)
	gamma = np.random.uniform(0.7, 1.5, bs)
	delta = np.random.uniform(-1 , 1, bs)
	chroI_0 = tf.stack([tf.image.adjust_brightness(
						tf.image.adjust_gamma(
						  tf.image.adjust_hue(
							img, hue[i]), gamma[i]), delta[i])
							  for img, i in zip(tf.unstack(imgs_0), range(bs))])

	chroI_1 = tf.stack([tf.image.adjust_brightness(
						tf.image.adjust_gamma(
						  tf.image.adjust_hue(
							img, hue[i]), gamma[i]), delta[i])
							  for img, i in zip(tf.unstack(imgs_1), range(bs))])

	# Image / Flow Summary
	# image_summary(chroI_0, chroI_1, "D_chrom", None)
	return chroI_0, chroI_1

def rotation_crop(imgs_0, imgs_1, flows):
	# pretty ugly (TODO: check cv2 warp affine rotate)
	"""image rotation/scaling.
	Specifically we sample
	- translation from a the range [ 20%, 20%]
	of the image width for x and y;
	- rotation from [ -17 , 17 ];
	- scaling from [0.9, 2.0].
	"""
	bs = FLAGS.batchsize
	h, w = FLAGS.img_shape[:2]

	#- rotation from [ -17 , 17 ];
	angles = np.random.uniform(-FLAGS.max_rotate_angle, FLAGS.max_rotate_angle, bs)
	imgs_0 = tf.contrib.image.rotate(imgs_0, angles)
	imgs_1 = tf.contrib.image.rotate(imgs_1, angles)
	flows = tf.contrib.image.rotate(flows, angles)
	# check available area to crop out image according to random rotation
	diff = w-h
	hw_ratio = float(h)/w
	#tf.summary.image("Rotation_", imgs_0, 4)
	scales = []
	boxes = []
	# rotate image and crop out black borders
	# http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
	for ang in angles:
		quadrant = int(math.floor(ang / (math.pi / 2))) & 3
		sign_alpha = ang if ((quadrant & 1) == 0) else math.pi - ang
		alpha = (sign_alpha % math.pi + math.pi) % math.pi
		bb_w = w * math.cos(alpha) + h * math.sin(alpha)
		bb_h = w * math.sin(alpha) + h * math.cos(alpha)
		gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)
		delta = math.pi - alpha - gamma
		length = h if (w < h) else w
		d = length * math.cos(alpha)
		a = d * math.sin(alpha) / math.sin(delta)
		y = a * math.cos(gamma)
		x = y * math.tan(gamma)
		max_width = bb_w - 2 * x
		max_height =bb_h - 2 * y
		# these are the maximum widths/ heights given the rotation from angle ang
		# normalized coordinates
		scale = max_width/w
		x1 = (1 - scale)/2
		x2 = 1 - x1
		y1 = (1 - scale)/2
		y2 = 1 - y1
		scales.append(scale)
		# if rotation forces scale to be already bigger than 2 do nothing (Never happening here)
		if scale <= 0.5:
			boxes.append([x1, y1, x2, y2])
		else:
			# random choose scale smaller than 2 and
			# random choose box to crop from the window given by roation -> random translation
			# new scale has to big enough to cut rotation error out
			new_scale = np.random.uniform(0.5, scale)
			new_width = w*new_scale
			new_height = h*new_scale
			x1_s = x1 + np.random.uniform(0, 1- new_width/max_width)
			x2_s = min(x1_s + new_width/w, x2)
			y1_s = y1 + np.random.uniform(0, 1- new_height/max_height)
			y2_s =  min(y1_s + new_height/h, y2)
			boxes.append([x1_s, y1_s, x2_s, y2_s])

	crop_size = [h, w]
	box_ind = [i for i in range(bs)]
	rotI_0 = tf.image.crop_and_resize(imgs_0, boxes, box_ind, crop_size)
	rotI_1 = tf.image.crop_and_resize(imgs_1, boxes, box_ind, crop_size)
	rotF = tf.image.crop_and_resize(flows, boxes, box_ind, crop_size)
	# Image / Flow Summary
	# image_summary(rotI_0, rotI_1, "C_rotation", rotF)
	return rotI_0, rotI_1, rotF

def flows_to_img(flows):
	""" Pyfunc wrapper for flow to rgb trafo """

	def _flow_transform(flows):
		""" Transorm Cartesian Flow to rgb flow image for visualisation """

		flow_imgs = []
		for flow in flows:
			img = computeColor.computeImg(flow)
			# cv2 returns bgr images
			b,g,r = cv2.split(img)
			img = cv2.merge((r,g,b))
			flow_imgs.append(img)
		return [flow_imgs]

	flow_imgs = tf.py_func( _flow_transform, [flows],
					 [tf.uint8], stateful = False, name='flow_transform')

	flow_imgs = tf.squeeze(tf.stack(flow_imgs))
	flow_imgs.set_shape([FLAGS.batchsize] + FLAGS.img_shape)
	return flow_imgs

def bil_solv_var(img, flow, confidence, flow_gt):
	"""Pyfunc wrapper for the bilateral solver"""

	def _bil_solv_2(img, flow, conf, flow_gt):
		"""bilateral solver"""

		solved_flow = bills.flow_solver_flo_var(img, flow, conf,
										ast.literal_eval(FLAGS.grid_params), ast.literal_eval(FLAGS.bs_params))

		if FLAGS.write_flows:
			directory = FLAGS.logdir + FLAGS.dataset
			if not os.path.exists(directory):
				os.makedirs(directory)

			print("Writing:", directory + "/flow_" + "%04d.png" % FLAGS.flow_int)
			# save img
			imsave(directory + "/img_" + "%04d.png" % FLAGS.flow_int, img)

			# save flow img and .flo file
			flow_img = computeColor.computeImg(flow)
			b,g,r = cv2.split(flow_img)
			flow_img = cv2.merge((r,g,b))

			imsave(directory + "/flow_" + "%04d.png" % FLAGS.flow_int, flow_img)
			writeFlowFile.write(flow, directory + "/flow_" + "%04d.flo" % FLAGS.flow_int)

			# save solved flow image
			"""img = computeColor.computeImg(solved_flow)
			b,g,r = cv2.split(img)
			img = cv2.merge((r,g,b))
			imsave(directory + "/flow_solved_" + "%04d.png" % FLAGS.flow_int, img)
			writeFlowFile.write(solved_flow, directory + "/flow_solved_" + "%03d.flo" % FLAGS.flow_int)
			"""

			# save confidence image + ground truth .flo
			imsave(directory + "/confidence_" + "%04d.png" % FLAGS.flow_int, conf)
			writeFlowFile.write(flow_gt, directory + "/flow_gt_" + "%04d.flo" % FLAGS.flow_int)


		if FLAGS.flow_int == FLAGS.testsize:
		    FLAGS.flow_int = 1
		else:
		    FLAGS.flow_int += 1
		return solved_flow

	#print(img, flow, confidence)
	solved_flow = tf.py_func( _bil_solv_2, [img, flow, confidence, flow_gt],
							[tf.float32], name='bil_solv')
	solved_flow = tf.squeeze(tf.stack(solved_flow))
	solved_flow.set_shape(list(FLAGS.flow_shape))

	return solved_flow

def image_summary(imgs_0, imgs_1, text, flows):
	""" Write image summary for tensorboard / data augmenation """
	if FLAGS.imgsummary:
		if imgs_0 != None and imgs_1 != None:
			tf.summary.image(text + "_img_0", imgs_0, FLAGS.img_summary_num)
			tf.summary.image(text + "_img_1", imgs_1, FLAGS.img_summary_num)
		if flows != None:
			flow_imgs = flows_to_img(flows)
			tf.summary.image(text + "_flow", flow_imgs, FLAGS.img_summary_num)

def create_train_op(global_step):
 	"""Sets up the training Ops."""

	slim.model_analyzer.analyze_vars(
		tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), print_info=True)

	learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), FLAGS.boundaries, FLAGS.values, name=None)
	train_loss = tf.losses.get_total_loss()
	tf.summary.scalar('Learning_Rate', learning_rate)
	tf.summary.scalar('Training Loss', train_loss)

	trainer = tf.train.AdamOptimizer(learning_rate)
	train_op = slim.learning.create_train_op(train_loss, trainer)
	return train_op
