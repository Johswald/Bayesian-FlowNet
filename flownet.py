# FlowNet in Tensorflow
# ==============================================================================

import gzip
import os
import sys
import cv2

import psutil
import numpy as np
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim

import flownet_input

FLAGS = tf.app.flags.FLAGS

def inference(images_0, images_1, img_shape):
  """Build the flownet model up to where it may be used for inference.
  """
  net = tf.concat([images_0, images_1], img_shape[-1],  name='concat_0')
  # stack of convolutions
  convs = {"conv1" : [64, [7,7], 2],
          "conv2_1" : [128, [5,5], 2], 
          "conv3" : [256, [5,5], 2], 
          "conv3_1" : [256, [3,3], 1], 
          "conv4" : [512, [3,3], 2], 
          "conv4_1" : [512, [3,3], 1], 
          "conv5" : [512, [3,3], 2], 
          "conv5_1" : [512, [3,3], 1],
          "conv6" : [1024, [3,3], 2], 
          "conv6_1" : [1024, [3,3], 1], 
      }

  with slim.arg_scope([slim.conv2d], padding='SAME',
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    #convolutions
    for key in sorted(convs): 
      net = slim.conv2d(net, convs[key][0], convs[key][1], convs[key][2], scope=key)
    # deconv + flow
    for i in range(4):
      # flow predict + flow deconv
      flow_predict = slim.conv2d(net, 2, [3, 3], 1, scope='flow_' + str(6-i))
      flow_up = slim.conv2d_transpose(flow_predict, 2, [4, 4], 2, scope='flow_dv_'+ str(6-i))
      # devonv net + concat
      deconv = slim.conv2d_transpose(net, 512/2**i , [4, 4], 2, scope='deconv_'+ str(5-i))
      # get old convolution
      to_concat = tf.get_default_graph().get_tensor_by_name('conv'+str(5-i)+'_1/Relu:0')
      net = tf.concat([deconv, to_concat, flow_up], img_shape[-1], name='concat_' + str(i+1))
    flow_predict = slim.conv2d(net, 2, [3, 3], 1, scope='flow_pred')
  # bilinear upsample? (TODO)
  flow_up = tf.image.resize_images(flow_predict, img_shape[:2])
  return flow_up

def _affine_transform(imgs_0, imgs_1, flows, shape):
  """Affine Transformation with OpenCV help (warpAffine)"""
  bs = FLAGS.batch_size
  h, w = shape[:2]
  c = np.float32([w, h]) / 2.0
  mat = np.random.normal(size=[bs, 2, 3])
  mat[:, :2, :2] = mat[:, :2, :2] * 0.2 + np.eye(2)
  mat[:, :, 2] = mat[:, :, 2] * 0.8 + c - mat[:, :2, :2].dot(c)

  for mat_i, img_0, img_1, flow, i in zip(mat, imgs_0, imgs_1, flows, range(bs)):
    """
    if np.amax(img_0) > 1 or np.amax(img_1) > 1 or np.amin(img_0) < 0 or np.amin(img_0) < 0:
      print(np.amin(img_0), np.amin(img_1), np.amax(img_0), np.amax(img_1))
      exit()"""
    aug_0 = cv2.warpAffine(img_0, mat_i, (w, h), borderMode=3)
    aug_1 = cv2.warpAffine(img_1, mat_i, (w, h), borderMode=3)
    aug_f = cv2.warpAffine(flow, mat_i, (w, h), borderMode=3)
    if np.random.rand() > 0.8:
      aug_0 = cv2.GaussianBlur(aug_0, (7, 7), -1)
      aug_1 = cv2.GaussianBlur(aug_1, (7, 7), -1)
      aug_f = cv2.GaussianBlur(aug_f, (7, 7), -1)
    imgs_0[i] = aug_0
    imgs_1[i] = aug_1
    flows[i] = aug_f
  #print("IMAGE", imgs_0[0])
  return [imgs_0, imgs_1, flows]

def affine_trafo(data, img_shape, flow_shape):
  """affine transformation """
  aug_data = tf.py_func( _affine_transform, [data[0], data[1], data[2], img_shape], 
              [tf.float32, tf.float32, tf.float32], name='affine_transform')
  # does a more elegant way exist?
  augI_0 =  aug_data[0]
  augI_1 = aug_data[1]
  augF =  aug_data[2]
  augI_0.set_shape([FLAGS.batch_size] + list(img_shape))
  augI_1.set_shape([FLAGS.batch_size] + list(img_shape))
  augF.set_shape([FLAGS.batch_size] + list(flow_shape))
  return augI_0, augI_1, augF

def affine_trafo_2(imgs_0, imgs_1, flows, shape):
  """
  bs = FLAGS.batch_size
  h, w = shape[:2]
  c = np.float32([w, h]) / 2.0
  mat = np.random.normal(size=[bs, 2, 3])
  mat[:, :2, :2] = mat[:, :2, :2] * 0.2 + np.eye(2)
  mat[:, :, 2] = mat[:, :, 2] * 0.8 + c - mat[:, :2, :2].dot(c)"""
  #print(FLAGS.mat)
  bs = FLAGS.batch_size
  h, w = shape[:2]
  c = np.float32([w, h]) / 2.0
  mat = np.random.normal(size=[bs, 2, 3])
  mat[:, :2, :2] = mat[:, :2, :2] * 0.2 + np.eye(2)
  mat[:, :, 2] = mat[:, :, 2] * 0.8 + c - mat[:, :2, :2].dot(c)

  flip_x = [np.random.choice([1, -1]).tolist() for i in range(bs)]
  flip_y = [np.random.choice([1, -1]).tolist() for i in range(bs)]
  mat[:, 0, 0] = 0
  mat[:, 1, 1] = 0
  mat[:, 0, 2] = 10
  mat[:, 1, 2] = 10
  mat[:, 1, 0] = 0
  mat[:, 0, 1] = 0

  reshape = [np.append(np.reshape(ma,[6]), [0,0]).tolist() for ma in mat]
  print(reshape)
  imgs_0 = tf.contrib.image.transform(imgs_0, reshape)
  imgs_1 = tf.contrib.image.transform(imgs_1, reshape)
  flows = tf.contrib.image.transform(flows, reshape)

  return imgs_0, imgs_1, flows

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
  bs = FLAGS.batch_size

  # multiplicative color changes to the RGB channels per image from [0.5, 2]; 
  # 1. Own testet replacement with saturation / hue
  # 2. gamma values from [0.7, 1.5] and 
  # 3. Own testet brightness changes
  # different transformation in batch

  hue = np.random.uniform(-1, 1, bs)
  gamma = np.random.uniform(0.7, 1.5, bs)
  delta = np.random.uniform(-1 , 1, bs)
  imgs_0 = tf.stack([tf.image.adjust_brightness(
                        tf.image.adjust_gamma(
                          tf.image.adjust_hue(
                            img, hue[i]), gamma[i]), delta[i])
                              for img, i in zip(tf.unstack(imgs_0), range(bs))])

  imgs_1 = tf.stack([tf.image.adjust_brightness(
                        tf.image.adjust_gamma(
                          tf.image.adjust_hue(
                            img, hue[i]), gamma[i]), delta[i])
                              for img, i in zip(tf.unstack(imgs_1), range(bs))])

  return imgs_0, imgs_1

def rotation(imgs_0, imgs_1, flows, shape):
  """image rotation/scaling. 
  Specifically we sample 
  - translation from a the range [ 20%, 20%] 
  of the image width for x and y; 
  - rotation from [ -17 , 17 ]; 
  - scaling from [0.9, 2.0]. 
  """
  bs = FLAGS.batch_size
  h, w = shape[:2]

  #- rotation from [ -17 , 17 ]; 
  angles = np.random.uniform(-0.17, 0.17, bs)
  imgs_0 = tf.contrib.image.rotate(imgs_0, angles)
  imgs_1 = tf.contrib.image.rotate(imgs_1, angles)
  flows = tf.contrib.image.rotate(flows, angles)
  # check available area to crop out image according to random rotation
  diff = w-h
  hw_ratio = float(h)/w
  #tf.summary.image("Rotation_", imgs_0, 4)
  scales = []
  boxes = []
  # get scale needed to undo rotation mistake
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
  box_ind = [i for i in range(8)]
  imgs_0 = tf.image.crop_and_resize(imgs_0, boxes, box_ind, crop_size)
  imgs_1 = tf.image.crop_and_resize(imgs_1, boxes, box_ind, crop_size)
  flows = tf.image.crop_and_resize(flows, boxes, box_ind, crop_size)

  #tf.summary.image("crop_and_resize" , imgs_0, 8)

  #imgs0_test = tf.stack([tf.image.resize_images(tf.image.central_crop(img, scales[i]), shape[:2]) 
  #                      for img, i in zip(tf.unstack(imgs_0), range(bs))])

  #tf.summary.image("Rotation_crop" , imgs_0, 2)
  return imgs_0, imgs_1, flows

def _hsv_transform(flows, shape):
  """Transorm Cartesian Flow to HSV flow for visualisation"""
  """TODO: this is not correct, copy official matlab version"""
  hsv_flows = []
  h = shape[0]
  w = shape[1]
  for flow in flows:
    a = flow[: , : , 0]
    b = flow[: , : , 1]
    mag, ang = cv2.cartToPolar(a, b)
    #http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    hsv = np.zeros((int(h), int(w), 3), np.uint8)
    hsv[:, :, 0] = ang * 180 / np.pi / 2
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    hsv_flows.append(hsv)
  return [hsv_flows]

def flows_to_hsv(flows, flow_shape):
  """ Pyfunc wrapper for flow to hsv trafo """
  hsv_flows = tf.py_func( _hsv_transform, [flows, flow_shape], 
                         [tf.uint8], name='hsv_transform')[0]
  hsv_flows.set_shape([FLAGS.batch_size] + list(flow_shape))
  return hsv_flows

def loss(calc_flows, flows, flow_shape):
  """
  loss on the aee (average endpoint error)
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3478865/
  loss on chairs set: 2.71"""
  # Is this correct? Could be absolute value?
  abs_loss = tf.reduce_sum(tf.abs(tf.subtract(calc_flows,flows)))
  return abs_loss/(FLAGS.batch_size*flow_shape[0]*flow_shape[1])

def training(loss, global_step, learning_rate):

  """Sets up the training Ops."""
  """For training CNNs we use a modified version of the caffe [20] framework. 
  We choose Adam [22] as optimization method because for our task it shows faster 
  convergence than standard stochastic gradient descent with momentum. We fix the 
  parameters of Adam as recommended in [22]: B1 = 0.9 and B2 = 0.999. 
  """# Add a scalar summary for the snapshot loss.
  # Create the gradient descent optimizer with the given learning rate.
  tf.summary.scalar('Training Loss', loss)
  optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate, beta1=0.9, 
    beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def read_data_lists():
  """Construct data lists with batch reader function.
  ....
  """
  return flownet_input.read_data_lists(FLAGS.data_dir, FLAGS.split_list)


