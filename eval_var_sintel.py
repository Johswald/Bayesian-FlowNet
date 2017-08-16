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
import bilateral_solver as bs
import computeColor
import writeFlowFile

from skimage.io import imread

dir_path = dirname(os.path.realpath(__file__))

# Basic model parameters as external flags.
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', '/sintel', 'Data Set in use')

flags.DEFINE_string('grid_params', "{ 'sigma_luma' : 2,'sigma_chroma': 4, 'sigma_spatial': 2}",
                    'grid_params')

flags.DEFINE_string('bs_params', "{'lam': 30, 'A_diag_min': 1e-5, 'cg_tol': 1e-5, 'cg_maxiter': 25}",
                    'bs_params')

flags.DEFINE_integer('batchsize', 20, 'Batch size for eval loop.')

flags.DEFINE_integer('eval_interval_secs', 300,
                     'How many seconds between executions of the eval loop.')

flags.DEFINE_integer('testsize', 1041,
                     'Number of test samples')

flags.DEFINE_integer('record_bytes', 3571724, 'Size of .flo bytes')

flags.DEFINE_integer('d_shape_img', [436, 1024, 3],
                           'Data shape: width, height, channels')

flags.DEFINE_integer('d_shape_flow', [436, 1024, 2],
                           'Data shape: width, height, channels')

flags.DEFINE_integer('img_shape', [384, 512, 3],
                           'Image shape: width, height, channels')

flags.DEFINE_integer('flow_shape', [384, 512, 2],
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
	x = tf.slice(vec,[0, 0, 0, 0], [size] + FLAGS.img_shape[:2] + [1])
	y = tf.slice(vec,[0, 0, 0, 1], [size] + FLAGS.img_shape[:2] + [1])
	return tf.squeeze(x), tf.squeeze(y)

def aee_f(flows, calc_flows, size):
    square = tf.square(flows - calc_flows)
    x , y = slice_vector(square, size)
    sqr = tf.sqrt(tf.add(x, y))
    aee = tf.metrics.mean(sqr)
    return aee

def var_mean_2(flow_to_mean):
    """Pyfunc wrapper for the bilateral solver"""

    def _var_mean_2(flow_to_mean):
        """bilateral solver"""
        flow_to_mean = np.array(flow_to_mean)
        x = flow_to_mean[:,:,:,0]
        y = flow_to_mean[:,:,:,1]
        var_x = np.var(x, 0)
        var_y = np.var(y, 0)
        #var_mea = np.mean(np.array([var_x, var_y]), 0)
        var_mea = (var_x + var_y)/2
        # TODO check /2, /8 here ...
        var_mea = np.exp(-1*np.array(var_mea, np.float32)/8)
        print(np.amax(var_mea), np.amin(var_mea))
        # normalize
        #print(np.amax(var_mea), np.amin(var_mea))
        #print(np.amax(var_mea), np.amin(var_mea))
        flow_x_m = np.mean(x, 0)
        flow_y_m = np.mean(y, 0)
        flow_to_mean = np.zeros(list(FLAGS.flow_shape), np.float32)
        flow_to_mean[:,:,0] = flow_x_m
        flow_to_mean[:,:,1] = flow_y_m
        var_img = np.zeros(list(FLAGS.img_shape), np.float32)
        var_img[:,:,0] = var_mea
        var_img[:,:,1] = var_mea
        var_img[:,:,2] = var_mea
        return [flow_to_mean, var_mea, var_img]

    solved_data = tf.py_func( _var_mean_2, [flow_to_mean], [tf.float32, tf.float32, tf.float32], name='flow_mean')
    mean, var, var_img = solved_data[:]
    mean = tf.squeeze(tf.stack(mean))
    var = tf.squeeze(tf.stack(var))
    var_img = tf.squeeze(tf.stack(var_img))
    mean.set_shape(list(FLAGS.flow_shape))
    var.set_shape(list(FLAGS.flow_shape[:2])+ [1])
    var_img.set_shape(list(FLAGS.img_shape))
    return mean, var, var_img

def main(_):
    """Evaluate FlowNet for test set"""

    with tf.Graph().as_default():
        # Generate tensors from numpy images and flows.
        var_num = 1
        img_0, img_1, flow = flownet_tools.get_data_sintel(FLAGS.datadir, False, var_num)

        #resize
        img_0 = tf.image.resize_nearest_neighbor(img_0, FLAGS.img_shape[:2])
        img_1 = tf.image.resize_nearest_neighbor(img_1, FLAGS.img_shape[:2])
        flow = tf.image.resize_nearest_neighbor(flow, FLAGS.flow_shape[:2])

        imgs_0 = tf.squeeze(tf.stack([img_0 for i in range(FLAGS.batchsize)]))
        imgs_1 = tf.squeeze(tf.stack([img_1 for i in range(FLAGS.batchsize)]))
        flows = tf.squeeze(tf.stack([flow for i in range(FLAGS.batchsize)]))

        # img summary after loading
        flownet.image_summary(imgs_0, imgs_1, "A_input", flows)

        # Get flow tensor from flownet model
        calc_flows = architectures.flownet_dropout(imgs_0, imgs_1, flows)
        
        flow_mean, confidence, conf_img  = var_mean_2(calc_flows)

        #confidence = tf.image.convert_image_dtype(confidence, tf.uint16)
        # calc EPE / AEE = ((x1-x2)^2 + (y1-y2)^2)^1/2
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3478865/

        aee = aee_f(flow, flow_mean, var_num)
        # bilateral solver
        img_0 = tf.squeeze(tf.stack(img_0))
        flow_s = tf.squeeze(tf.stack(flow))
        solved_flow = flownet.bil_solv_var(img_0, flow_mean, confidence, flow_s)
        aee_bs = aee_f(flow, solved_flow, var_num)

        metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
          	      "AEE": slim.metrics.streaming_mean(aee),
                  "AEE_BS": slim.metrics.streaming_mean(aee_bs),
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
        num_batches = math.ceil(FLAGS.testsize)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True

        slim.evaluation.evaluation_loop(
        	master=FLAGS.master,
            checkpoint_dir=FLAGS.logdir + '/train',
            logdir=FLAGS.logdir + '/eval_var_sintel',
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
      default='data/Sintel/training/',
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
