"""
Definitions and utilities for the FlowNet model
This file contains functions for data augmentation, summary and training ops
"""

import os
import math
import time

import cv2
import numpy as np
from numpy import linalg as LA
from skimage.io import imsave
from PIL import Image
from PIL.ImageEnhance import *

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import flags
from tensorflow.contrib.keras.python.keras import backend as K

import computeColor
import bilateral_solver as bils
import writeFlowFile

FLAGS = flags.FLAGS

flags.DEFINE_float('max_rotate_angle', 17.0,
                   'Mxx rotation angle')


def adjust_gamma(image, gamma=1.0):
    """Adjust gamma value of image

	Keyword arguments:
	imgs_0 -- first image of image pair (with length of bath size)
	gamma -- gamma value (default = 1.0)
    """

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def chromatic_augm(imgs_0, imgs_1):
    """ (Slow) chromatic augmenation discibed by FlowNet paper
	(https://arxiv.org/pdf/1504.06852.pdf)
    - The Gaussian noise has a sigma uniformly sampled from [0, 0.04]
    - contrast is sampled within [-0.8, 0.4];
    - multiplicative color changes to the RGB channels per image from [0.5, 2];
    - gamma values from [0.7, 1.5] and
    - additive brightness changes using Gaussian with a sigma of 0.2.

	Keyword arguments:
	imgs_0 -- first image of image pair (with length of bath size)
	imgs_1 -- second image of image pair (with length of bath size)
    """

    def _chromatic(imgs_0, imgs_1):
        """Tensorflow Pyfunc for chromatic augmenation"""
        bs = FLAGS.batchsize
        contrast = np.array([np.random.uniform(0.2, 1.4)
                             for i in range(bs)]).astype(np.float32)
        color = np.array([np.random.uniform(0.5, 2)
                          for i in range(bs)]).astype(np.float32)
        gamma = np.array([np.random.uniform(0.7, 1.5)
                          for i in range(bs)]).astype(np.float32)
        brightness = np.array([np.random.normal(1, 0.2)
                               for i in range(bs)]).astype(np.float32)
        noise = np.array([np.random.uniform(0, 0.04)
                          for i in range(bs)]).astype(np.float32)
        for img_0, img_1, i in zip(imgs_0, imgs_1, range(bs)):
            img_0 = Contrast(Image.fromarray(img_0)).enhance(contrast[i])
            img_1 = Contrast(Image.fromarray(img_1)).enhance(contrast[i])
            img_0 = Color(img_0).enhance(color[i])
            img_1 = Color(img_1).enhance(color[i])
            img_0 = Brightness(img_0).enhance(brightness[i])
            img_1 = Brightness(img_1).enhance(brightness[i])
            img_0 = np.array(img_0, np.uint8)
            img_1 = np.array(img_1, np.uint8)
            img_0 = adjust_gamma(img_0, gamma[i]).astype(np.float32)
            img_1 = adjust_gamma(img_1, gamma[i]).astype(np.float32)
            gaussian_noise = np.random.normal(0, noise[i], img_0.shape) * 255.0
            imgs_0[i] = np.maximum(0, np.minimum(
                img_0 + gaussian_noise, 255.0))
            imgs_1[i] = np.maximum(0, np.minimum(
                img_1 + gaussian_noise, 255.0))
        return [imgs_0.astype(np.float32), imgs_1.astype(np.float32)]

    chromatic_data = tf.py_func(_chromatic, [imgs_0, imgs_1],
                                [tf.float32, tf.float32], name='chromatic_augm')
    imgs_0, imgs_1 = chromatic_data[:]
    imgs_0.set_shape([FLAGS.batchsize] + list(FLAGS.img_shape))
    imgs_1.set_shape([FLAGS.batchsize] + list(FLAGS.img_shape))
    return imgs_0, imgs_1


def fast_chromatic_augm(imgs_0, imgs_1):
    """ (Fast) tensorflow chromatic augmenation

	Keyword arguments:
	imgs_0 -- first image of image pair (with length of bath size)
	imgs_1 -- second image of image pair (with length of bath size)
    """

    bs = FLAGS.batchsize

    def _get_randoms(bs):
        """Tensorflow Pyfunc to get random numbers for chromatic augmentation"""
        hue = np.random.uniform(-0.3, 0.3, bs).astype(np.float32)
        bright = np.random.uniform(- 26. / 255.,
                                   26. / 255., bs).astype(np.float32)
        satur = np.random.uniform(0.7, 1.3, bs).astype(np.float32)
        cont = np.random.uniform(0.7, 1.3, bs).astype(np.float32)
        return [hue, bright, satur, cont]

    hue, bright, satur, cont = tf.py_func(_get_randoms,
                                          [bs], [tf.float32, tf.float32, tf.float32, tf.float32], name='get_randoms')[:]

    hue = tf.stack(hue)
    hue.set_shape([bs])

    satur = tf.stack(satur)
    satur.set_shape([bs])

    cont = tf.stack(cont)
    cont.set_shape([bs])

    bright = tf.stack(bright)
    bright.set_shape([bs])

    imgs_0 = tf.stack([tf.image.adjust_hue(
        tf.image.adjust_saturation(
            tf.image.adjust_contrast(
                tf.image.adjust_brightness(
                    img, h), s), c), b)
        for img, h, s, c, b in zip(tf.unstack(imgs_0),
                                   tf.unstack(hue),
                                   tf.unstack(satur),
                                   tf.unstack(cont),
                                   tf.unstack(bright))])

    imgs_1 = tf.stack([tf.image.adjust_hue(
        tf.image.adjust_saturation(
            tf.image.adjust_contrast(
                tf.image.adjust_brightness(
                    img, h), s), c), b)
        for img, h, s, c, b in zip(tf.unstack(imgs_1),
                                   tf.unstack(hue),
                                   tf.unstack(satur),
                                   tf.unstack(cont),
                                   tf.unstack(bright))])

    # Image / Flow Summary
    # image_summary(imgs_0, imgs_1, "D_chrom", None)
    return imgs_0, imgs_1


def warp_image(img, flow):
    """ Use optical flow to backwarp image to the previous
	(https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/flowlib.py)

	Keyword arguments:
	img -- image to warp
	flow -- optical flow
    """

    image_height = img.shape[0]
    image_width = img.shape[1]
    warp = np.zeros(img.shape)

    for x in range(image_width):
        for y in range(image_height):
            fx = round(flow[x, y, 0] + x)
            fy = round(flow[x, y, 1] + y)
            if fx >= 0 and fy >= 0 and fx < image_width and fy < image_height:
                print(x, y, int(fx), int(fy))
               warp[x, y] = img[int(fx), int(fy)]
    return warp


def warp(imgs_0, imgs_1, flows, name):
    """ Use optical flow to warp image to the next
	(https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/flowlib.py)

	Keyword arguments:
    imgs_0 -- first image of image pair (with length of bath size)
    imgs_1 -- second image of image pair (with length of bath size)
    flows -- ground truth optical flows between imgs_0, imgs_1
	name -- naming for summary
    """

    def _warp(imgs_1, flows):
        """Tensorflow Pyfunc to backwarp image to the previous"""
        _, h, w, _ = imgs_1.shape
        for img, flow, i in zip(imgs_1, flows, range(FLAGS.batchsize)):
            imgs_1[i] = warp_image(img, flow)
        return imgs_1

    bs = FLAGS.batchsize
    warped_imgs = tf.py_func(_warp, [imgs_1, flows],
                             tf.float32, name='warp')
    warped_imgs = tf.stack(warped_imgs)
    warped_imgs.set_shape([bs] + list(FLAGS.img_shape))

    tf.summary.image("image_0" + name, imgs_0, FLAGS.img_summary_num)
    tf.summary.image("warped_0_to_1" + name,
                     warped_imgs, FLAGS.img_summary_num)
    tf.summary.image("image_1" + name, imgs_1, FLAGS.img_summary_num)


def rotation_crop_trans(rotI_0, rotI_1, rotF):
    """ Hacky random rotation, crop, translate, flip, resize of images, flows
    (https://arxiv.org/pdf/1504.06852.pdf)
    Specifically we sample
    - rotation from [ -17 , 17 ];
    - translation from a the range [20%, 20%] of the image still available after rot;
    - scaling maximum 2.
    - flipping is not used but could be used later

    Keyword arguments:
    rotI_0 -- first image of image pair (with length of bath size)
    rotI_1 -- second image of image pair (with length of bath size)
    rotF -- ground truth optical flows between imgs_0, imgs_1
    """

    warp(rotI_0, rotI_1, rotF, "_rotF_first")
    def _flip(rotI_0, rotI_1, rotF):
        """Tensorflow Pyfunc to flip randomly flip images/ flows"""
        for img0, img1, flow, i in zip(rotI_0, rotI_1, rotF, range(bs)):
            # with 50% change we flip horizontally / vertically
            # we also have to flip flow pointers as well
            # horizontal
            if np.random.uniform(0, 1) > 0.5:
                img0 = cv2.flip(img0, 0)
                img1 = cv2.flip(img1, 0)
                flow = cv2.flip(flow, 0)
                flow[:, :, 1] = flow[:, :, 1] * -1

            # vertical
            if np.random.uniform(1, 2) > 1.5:
                img0 = cv2.flip(img0, 1)
                img1 = cv2.flip(img1, 1)
                flow = cv2.flip(flow, 1)
                flow[:, :, 0] = flow[:, :, 0] * -1

            rotI_0[i] = img0
            rotI_1[i] = img1
            rotF[i] = flow

        return [rotI_0, rotI_1, rotF]

    def _randomize_rotcroptrans(rotF):
        """Tensorflow Pyfunc to build rotation + translation coordinates
        for later use by tensorflow rotate / crop_and_resize"""

        bs = FLAGS.batchsize
        h, w = FLAGS.img_shape[:2]

        #- rotation from [ -17 , 17 ];
        angles = np.random.uniform(-FLAGS.max_rotate_angle,
                                   FLAGS.max_rotate_angle, bs).astype(np.float32) * np.pi / 180.0

        # after that we randomly choose a new scale between this minimum and
        # a 50% maximum scale of original image
        # than we randomly choose to boxes to crop with tensorflow this new scale
        boxes_0 = []
        boxes_1 = []

        for flow, ang, i in zip(rotF, angles, range(bs)):

            ############################
            # CROP BOXES + TRANSLATION
            ############################

            # after we rotated images, flows we check available image size ( no black corners)
            # after that we randomly choose a new scale between this minimum and maximum
            # scale 50% of original image
            # than we randomly choose to boxes to crop with this new scale
            angle1_rad = np.pi / 2 - np.abs(ang % np.pi - np.pi / 2)
            c1 = np.cos(angle1_rad)
            s1 = np.sin(angle1_rad)
            c_diag = h / np.sqrt(h * h + w * w)
            s_diag = w / np.sqrt(h * h + w * w)
            # minimum scale
            # this is somehow not enough, we add 0.1
            # when testet on 17degree, there was always a small black corner <- rounding error?
            # angle 17 -> 0.085 fix or angle 8 -> 0.04 fix seems to work (half os dec. degree)
            scale = c_diag / (c1 * c_diag + s1 * s_diag) - \
                (abs(ang) / np.pi * 180.0) / 200

            # 1. random choose scale smaller than 2 - we choose less
            # 2. random choose 2 box to crop from the window given by roation -> random translation
            new_scale = np.random.uniform(scale - 0.2, scale)
            min_c = (1 - scale) / 2.
            max_c = (1 - new_scale) / 2.

            x1_0 = np.random.uniform(min_c, max_c)
            x1_1 = np.random.uniform(min_c, max_c)
            x2_0 = x1_0 + new_scale
            x2_1 = x1_1 + new_scale

            y1_0 = np.random.uniform(min_c, max_c)
            y1_1 = np.random.uniform(min_c, max_c)
            y2_0 = y1_0 + new_scale
            y2_1 = y1_1 + new_scale

            boxes_0.append([y1_0, x1_0, y2_0, x2_0])
            boxes_1.append([y1_1, x1_1, y2_1, x2_1])

            x = flow[:, :, 0]
            y = flow[:, :, 1]
            # rotate x, y flow pointers
            x = np.multiply(x, np.cos(ang)) + np.multiply(y, np.sin(ang))
            y = np.multiply(x, -np.sin(ang)) + np.multiply(y, np.cos(ang))

            # scale flow pointers
            x *= (1 / new_scale)
            y *= (1 / new_scale)

            # (scaled) translation
            flow[:, :, 0] = x + (x1_0 - x1_1) * w / new_scale
            flow[:, :, 1] = y + (y1_0 - y1_1) * h / new_scale

            rotF[i] = flow

        boxes_0 = np.array(boxes_0, np.float32)
        boxes_1 = np.array(boxes_1, np.float32)
        return [rotF, angles, boxes_0, boxes_1]

    bs = FLAGS.batchsize
    h, w = FLAGS.img_shape[:2]

    rotF, angles, boxes_0, boxes_1 = tf.py_func(_randomize_rotcroptrans,
                                                [rotF], [tf.float32, tf.float32, tf.float32, tf.float32], name='rotcroptrans')[:]

    rotF = tf.stack(rotF)
    rotF.set_shape([bs] + list(FLAGS.flow_shape))

    angles = tf.stack(angles)
    angles.set_shape([bs])

    boxes_0 = tf.stack(boxes_0)
    boxes_0.set_shape([bs, 4])

    boxes_1 = tf.stack(boxes_1)
    boxes_1.set_shape([bs, 4])

    ##############
    #   APPLY
    ##############

    # rotate all images and flow "picture"
    # tensorflow doesnt allow to choose bilinear here?
    rotI_0 = tf.contrib.image.rotate(rotI_0, angles)
    rotI_1 = tf.contrib.image.rotate(rotI_1, angles)
    rotF = tf.contrib.image.rotate(rotF, angles)

    crop_size = [h, w]
    box_ind = [i for i in range(bs)]

    # bilinear is default
    rotI_0 = tf.image.crop_and_resize(rotI_0, boxes_0, box_ind, crop_size)
    rotI_1 = tf.image.crop_and_resize(rotI_1, boxes_1, box_ind, crop_size)
    rotF = tf.image.crop_and_resize(rotF, boxes_0, box_ind, crop_size)

    ############################
    # FLIP IMGs + FLOW
    ############################

    """
	rotI_0, rotI_1, rotF  = tf.py_func( _flip, [rotI_0, rotI_1, rotF],
			[tf.float32,tf.float32, tf.float32], name='flip')[:]

	rotI_0 = tf.stack(rotI_0)
	rotI_0.set_shape([bs] + list(FLAGS.img_shape))

	rotI_1 = tf.stack(rotI_1)
	rotI_1.set_shape([bs] + list(FLAGS.img_shape))

	rotF = tf.stack(rotF)
	rotF.set_shape([bs] + list(FLAGS.flow_shape))
	"""

    # Check data augmentation
    warp(rotI_0, rotI_1, rotF, "_rotF_sec")
    return rotI_0, rotI_1, rotF


def flows_to_img(flows):
    """Pyfunc wrapper to transorm flow vectors in color coding"""

    def _flow_transform(flows):
        """ Tensorflow Pyfunc to transorm flow to color coding"""

        flow_imgs = []
        for flow in flows:
            img = computeColor.computeImg(flow)
            # cv2 returns bgr images
            b, g, r = cv2.split(img)
            img = cv2.merge((r, g, b))
            flow_imgs.append(img)
        return [flow_imgs]

    flow_imgs = tf.py_func(_flow_transform, [flows],
                           [tf.uint8], stateful=False, name='flow_transform')

    flow_imgs = tf.squeeze(tf.stack(flow_imgs))
    flow_imgs.set_shape([FLAGS.batchsize] + FLAGS.d_shape_img)
    return flow_imgs


def bil_solv_var(img, flow, conf_x, conf_y, flow_gt):
    """ Bilateral solver is used to smooth signals in range and dimensional space
	(https://arxiv.org/pdf/1511.03296.pdf)

	To regulate smoothing, pixel confidence of the prediction
	can be used to tune the amount of smoothing.

	Keyword arguments:
	img -- first image of image pair (with length of bath size)
	flow --
	conf_x -- confidence of flow prediction in x direction
	conf_y -- confidence of flow prediction in y direction
	flow_gt -- ground truth optical flows between imgs_0, imgs_1
    """

    def _bil_solv(img, flow, conf_x, conf_y):
        """Tensorflow Pyfunc to use the bilateral solver """

        solved_flow = bils.bil_solv_flo(img, flow, conf_x, conf_y,
                                        FLAGS.grid_params, FLAGS.bs_params)

        return solved_flow

    solved_flow = tf.py_func(_bil_solv, [img, flow, conf_x, conf_y],
                             [tf.float32], name='bil_solv')
    solved_flow = tf.squeeze(tf.stack(solved_flow))
    solved_flow.set_shape(list(FLAGS.d_shape_flow))

    return solved_flow


def image_summary(imgs_0, imgs_1, text, flows):
    """ Write image summary for tensorboard / data augmenation """

    if FLAGS.imgsummary:
        if imgs_0 != None and imgs_1 != None:
            tf.summary.image(text + "_img_0", imgs_0, FLAGS.img_summary_num)
            tf.summary.image(text + "_img_1", imgs_1, FLAGS.img_summary_num)
        if flows != None:
			# transform flows to color codings
            flow_imgs = flows_to_img(flows)
            tf.summary.image(text + "_flow", flow_imgs, FLAGS.img_summary_num)

def create_train_op(global_step):
    """ Sets up the training ops. """

    # slim.model_analyzer.analyze_vars(
    #	tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), print_info=True)

    learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), FLAGS.boundaries,
                                                FLAGS.values, name=None)
    train_loss = tf.losses.get_total_loss()

    tf.summary.scalar('Learning_Rate', learning_rate)
    tf.summary.scalar('Training L1 Loss', train_loss)

    trainer = tf.train.AdamOptimizer(learning_rate)
    train_op = slim.learning.create_train_op(train_loss, trainer)
    return train_op
