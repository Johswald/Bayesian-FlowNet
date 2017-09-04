# FlowNet in Tensorflow
# Evaluation of the Kitti Dataset
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
import bilateral_solver as bs
import computeColor
import writeFlowFile

from skimage.io import imread

dir_path = dirname(os.path.realpath(__file__))

# Basic model parameters as external flags.
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', '/kitti_2012', 'Data Set in use')

flags.DEFINE_string('grid_params', { 'sigma_luma' : 8,'sigma_chroma': 8, 'sigma_spatial': 8},
                    'grid_params')

flags.DEFINE_string('bs_params', {'lam': 80, 'A_diag_min': 1e-5, 'cg_tol': 1e-5, 'cg_maxiter': 25},
                    'bs_params')

flags.DEFINE_integer('batchsize', 20, 'Batch size for eval loop.')

flags.DEFINE_integer('eval_interval_secs', 300,
                     'How many seconds between executions of the eval loop.')

flags.DEFINE_integer('testsize', 194, # 194
                     'Number of test samples')

flags.DEFINE_integer('d_shape_img', [375, 1242, 3],
                           'Data shape: width, height, channels')

flags.DEFINE_integer('d_shape_flow', [375, 1242, 2],
                           'Data shape:     width, height, channels')

flags.DEFINE_integer('img_net_shape', [384, 512, 3],
                           'Image shape: width, height, channels')

flags.DEFINE_integer('flow_net_shape', [384, 512, 2],
                           'Image shape: width, height, 2')

flags.DEFINE_integer('drop_rate', 0.5,
                           'Dropout change')

flags.DEFINE_string('master', '',
                    'BNS name of the TensorFlow master to use.')

flags.DEFINE_integer('img_summary_num', 1,
                           'Number of images in summary')

flags.DEFINE_string('confidence_png', dir_path + "/bilateral_solver/",
                           'path to confidence image for bilateral solver')

flags.DEFINE_integer('flow_int', 1,
						'integer to write .flo files')

def slice_vector(vec, size):
	x = tf.slice(vec,[0, 0, 0, 0], [size] + FLAGS.d_shape_img[:2] + [1])
	y = tf.slice(vec,[0, 0, 0, 1], [size] + FLAGS.d_shape_img[:2] + [1])
	return tf.squeeze(x), tf.squeeze(y)

def aee_f(flow_2d, flow_3d, flow_mean):
    "Python wrapper for average end point error"

    def _aee_f(flow_2d, flow_3d, flow_mean):
        "average end point error"
        square = np.square(flow_2d - flow_mean)
        square[np.where(flow_3d[:, :, 2] == 0)] = 0
        x = square[:, :, :, 0]
        y = square[:, :, :, 1]
        sqr = np.sqrt(x + y)
        # not using np.mean here since we dont want to count no valid pixels (with error 0)
        aee = np.true_divide(sqr.sum(1),(sqr!=0).sum(1)).astype(np.float32)
        return aee

    return tf.py_func( _aee_f, [flow_2d, flow_3d, flow_mean], [tf.float32], name='flow_aae')

def var_mean(flow_to_mean):
    """Pyfunc wrapper for the confidence / mean calculation"""

    def _var_mean(flow_to_mean):
        """ confidence / mean calculation"""
        flow_to_mean = np.array(flow_to_mean)
        x = flow_to_mean[:,:,:,0]
        y = flow_to_mean[:,:,:,1]
        var_x = np.var(x, 0)
        var_y = np.var(y, 0)
        #var_mea = np.mean(np.array([var_x, var_y]), 0)
        var_mea = (var_x + var_y)/2
        # TODO check /2, /8 here ...
        var_mea = np.exp(-1*np.array(var_mea, np.float32)/8)
        # normalize
        #print(np.amax(var_mea), np.amin(var_mea))
        #print(np.amax(var_mea), np.amin(var_mea))
        flow_x_m = np.mean(x, 0)
        flow_y_m = np.mean(y, 0)
        flow_to_mean = np.zeros(list(FLAGS.d_shape_flow), np.float32)
        flow_to_mean[:,:,0] = flow_x_m
        flow_to_mean[:,:,1] = flow_y_m
        var_img = np.zeros(list(FLAGS.d_shape_img), np.float32)
        var_img[:,:,0] = var_mea
        var_img[:,:,1] = var_mea
        var_img[:,:,2] = var_mea
        return [flow_to_mean, var_mea, var_img]

    solved_data = tf.py_func( _var_mean, [flow_to_mean], [tf.float32, tf.float32, tf.float32], name='flow_mean')
    mean, var, var_img = solved_data[:]
    mean = tf.squeeze(tf.stack(mean))
    var = tf.squeeze(tf.stack(var))
    var_img = tf.squeeze(tf.stack(var_img))
    mean.set_shape(list(FLAGS.d_shape_flow))
    var.set_shape(list(FLAGS.d_shape_flow[:2])+ [1])
    var_img.set_shape(list(FLAGS.d_shape_img))
    return mean, var, var_img

def main(_):
    """Evaluate FlowNet for Kitti 2012 test set"""

    with tf.Graph().as_default():
        # Generate tensors from numpy images and flows.
        var_num = 1
        img_0, img_1, flow = flownet_tools.get_data_kitti(FLAGS.datadir, False, var_num)
        flow_2D = (tf.slice(flow, [0, 0, 0, 0], [1] + FLAGS.d_shape_img[:2] + [2]) - 2**15) / 64.0

        # stack
        imgs_0 = tf.squeeze(tf.stack([img_0 for i in range(FLAGS.batchsize)]))
        imgs_1 = tf.squeeze(tf.stack([img_1 for i in range(FLAGS.batchsize)]))
        flows_2D = tf.squeeze(tf.stack([flow_2D for i in range(FLAGS.batchsize)]))

        #resize
        img_0_rs = tf.image.resize_nearest_neighbor(imgs_0, FLAGS.img_net_shape[:2])
        img_1_rs = tf.image.resize_nearest_neighbor(imgs_1, FLAGS.img_net_shape[:2])
        flows_2D_rs = tf.image.resize_nearest_neighbor(flows_2D, FLAGS.flow_net_shape[:2])
        # img summary after loading
        flownet.image_summary(imgs_0, imgs_1, "A_input", flows_2D)

        # Get flow tensor from flownet model
        calc_flows = architectures.flownet_dropout(img_0_rs, img_1_rs, flows_2D)
        flow_mean, confidence, conf_img  = var_mean(calc_flows)

        #confidence = tf.image.convert_image_dtype(confidence, tf.uint16)
        # calc EPE / AEE = ((x1-x2)^2 + (y1-y2)^2)^1/2
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3478865/
        aee = aee_f(flow_2D, flow, flow_mean)

        # bilateral solver
        img_0 = tf.squeeze(tf.stack(img_0))
        flow_s = tf.squeeze(tf.stack(flow_2D))
        solved_flow = flownet.bil_solv_var(img_0, flow_mean, confidence, flow_s)
        aee_bs = aee_f(flow_2D, flow, solved_flow)

        metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
          	      "AEE": slim.metrics.streaming_mean(aee),
                  "AEE_BS": slim.metrics.streaming_mean(aee_bs),
                 #"AEE_BS_No_Confidence": slim.metrics.streaming_mean(aee_bs_c1),
        })


        for name, value in metrics_to_values.iteritems():
            		tf.summary.scalar(name, value)
        # Define the summaries to write:
        flownet.image_summary(None, None, "FlowNetS_no_mean", calc_flows)
        solved_flows = tf.squeeze(tf.stack([solved_flow for i in range(FLAGS.batchsize)]))
        flow_means = tf.squeeze(tf.stack([flow_mean for i in range(FLAGS.batchsize)]))
        conf_imgs = tf.squeeze(tf.stack([conf_img for i in range(FLAGS.batchsize)]))
        flownet.image_summary(None, None, "FlowNetS BS", solved_flows)
        flownet.image_summary(None, None, "FlowNetS Mean", flow_means)
        flownet.image_summary(conf_imgs, conf_imgs, "Confidence", None)

        # Run the actual evaluation loop.
        num_batches = FLAGS.testsize

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True

        slim.evaluation.evaluation_loop(
        	master=FLAGS.master,
            checkpoint_dir=FLAGS.logdir + '/train',
            logdir=FLAGS.logdir + '/eval_var_kitti_2012',
            num_evals=num_batches,
            eval_op=metrics_to_updates.values(),
            eval_interval_secs=FLAGS.eval_interval_secs,
            summary_op=tf.summary.merge_all(),
            session_config=config,
            timeout=60*60
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--datadir',
      type=str,
      default='data/Kitti/2012/training/',
      help='Directory to put the input data.'
    )
    parser.add_argument(
      '--logdir',
      type=str,
      default='log_drop',
      help='Directory where to write event logs and checkpoints'
    )
    parser.add_argument(
      '--imgsummary',
      type=bool,
      default=True,
      help='Make image summary'
    )
    parser.add_argument(
        '--weights_reg',
        type=float,
        default=0,
        help='weights regularizer'
    )
    FLAGS.datadir = os.path.join(dir_path,  parser.parse_args().datadir)
    FLAGS.logdir = os.path.join(dir_path, parser.parse_args().logdir)
    FLAGS.imgsummary = parser.parse_args().imgsummary

    if parser.parse_args().weights_reg != 0:
        FLAGS.weights_reg = slim.l2_regularizer(args.weights_reg)
    else:
        FLAGS.weights_reg = None
    tf.app.run()
