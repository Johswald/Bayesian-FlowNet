# FlowNet in Tensorflow
# ==============================================================================

import cv2
import numpy as np
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import flags

import computeColor
import flownet_tools

FLAGS = flags.FLAGS

def inference(images_0, images_1):
  """Build the flownet model up to where it may be used for inference.
  """
  net = tf.concat([images_0, images_1], FLAGS.img_shape[-1],  name='concat_0')
  # stack of convolutions
  convs = {"conv1" : [64, [7,7], 2],
           "conv2_1" : [128, [5,5], 2], # _1 to concat easily later
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
      net = tf.concat([deconv, to_concat, flow_up], FLAGS.img_shape[-1], name='concat_' + str(i+1))
    flow_predict = slim.conv2d(net, 2, [3, 3], 1, scope='flow_pred')
  # resize  with ResizeMethod.BILINEAR as default
  flow_up = tf.image.resize_images(flow_predict, FLAGS.img_shape[:2])
  return flow_up

def _affine_transform(imgs_0, imgs_1, flows):
  """Affine Transformation with OpenCV help (warpAffine)"""
  bs = FLAGS.batchsize
  h, w, ch = FLAGS.img_shape
  c = np.float32([w, h]) / 2.0
  mat = np.random.normal(size=[bs, 2, 3])
  mat[:, :2, :2] = mat[:, :2, :2] * 0.2 + np.eye(2)
  mat[:, :, 2] = mat[:, :, 2] * 0.8 + c - mat[:, :2, :2].dot(c)

  for mat_i, img_0, img_1, flow, i in zip(mat, imgs_0, imgs_1, flows, range(bs)): 
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
  return [imgs_0, imgs_1, flows]

def affine_trafo(data):
  """affine transformation """
  shape = FLAGS.img_shape
  aug_data = tf.py_func( _affine_transform, [data[0], data[1], data[2]], 
              [tf.float32, tf.float32, tf.float32], name='affine_transform')
  augI_0, augI_1, augF = aug_data[:]
  augI_0.set_shape([FLAGS.batchsize] + list(FLAGS.img_shape))
  augI_1.set_shape([FLAGS.batchsize] + list(FLAGS.img_shape))
  augF.set_shape([FLAGS.batchsize] + list(FLAGS.flow_shape))
  
  # Image / Flow Summary
  image_summary(augI_0, augI_1, "C_affine", augF)
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
  image_summary(chroI_0, chroI_1, "B_chrom")

  return chroI_0, chroI_1

def rotation(imgs_0, imgs_1, flows):
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
  rotI_0 = tf.image.crop_and_resize(imgs_0, boxes, box_ind, crop_size)
  rotI_1 = tf.image.crop_and_resize(imgs_1, boxes, box_ind, crop_size)
  rotF = tf.image.crop_and_resize(flows, boxes, box_ind, crop_size)

  # Image / Flow Summary
  image_summary(rotI_0, rotI_1, "D_rotation", rotF)
  return rotI_0, rotI_1, rotF

def _flow_transform(flows):
  """ Transorm Cartesian Flow to rgb flow image for visualisation """

  flow_imgs = []
  h, w = FLAGS.flow_shape[:2]
  for flow in flows:
    img = computeColor.computeImg(flow)
    flow_imgs.append(img)
  return [flow_imgs]

def flows_to_img(flows):
  """ Pyfunc wrapper for flow to rgb trafo """

  flow_imgs = tf.py_func( _flow_transform, [flows], 
                         [tf.uint8], name='flow_transform')[0]

  flow_imgs.set_shape([FLAGS.batchsize] + list(FLAGS.flow_shape))
  return flow_imgs

def image_summary(imgs_0, imgs_1, text, flows=None):
  """ Write image summary for tensorboard / data augmenation """ 

  if FLAGS.imgsummary:
    tf.summary.image(text + "_img_0", imgs_0, FLAGS.img_summary_num)
    tf.summary.image(text + "_img_1", imgs_1, FLAGS.img_summary_num)
    if flows != None:
      flow_imgs = flows_to_img(flows)
      tf.summary.image(text + "_flow", flow_imgs, 2)

def train_loss(calc_flows, flows):
  """
  loss on the aee (average endpoint error)
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3478865/
  loss on chairs set: 2.71"""

  h, w = FLAGS.img_shape[:2]

  scale = 1/FLAGS.batchsize*h*w
  abs_loss = slim.losses.absolute_difference(calc_flows,flows)
  return abs_loss * scale

def create_train_op(train_loss, global_step):
  """Sets up the training Ops."""
  
  slim.model_analyzer.analyze_vars(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), print_info=True)

  learning_rate = tf.maximum(
                    tf.train.exponential_decay(
                        FLAGS.learning_rate,
                        global_step,
                        FLAGS.decay_steps,
                        FLAGS.decay_factor,
                        staircase=True),
                    FLAGS.minimum_learning_rate)

  tf.summary.scalar('Learning_Rate', learning_rate)
  tf.summary.scalar('Training_Loss', train_loss)

  trainer = tf.train.AdamOptimizer(learning_rate= learning_rate, beta1=0.9, 
                                    beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
  train_op = slim.learning.create_train_op(train_loss, trainer)
  return train_op

def read_data_lists():
  """Construct data lists with batch reader function.
  ....
  """
  return flownet_tools.read_data_lists(FLAGS.datadir, FLAGS.splitlist)