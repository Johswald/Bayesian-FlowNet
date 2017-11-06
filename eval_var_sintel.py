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
import bilateral_solver_var as bs
import computeColor
import writeFlowFile

from skimage.io import imread

dir_path = dirname(os.path.realpath(__file__))

# Basic model parameters as external flags.
FLAGS = flags.FLAGS

# ARCHITECTURE
model = architectures.flownet_s

# DATA
# [436, 1024, 3]

flags.DEFINE_integer('d_shape_img', [448, 1024, 3],
                           'Data shape: width, height, channels')

flags.DEFINE_integer('d_shape_flow', [448, 1024, 2],
                           'Data shape: width, height, channels')

flags.DEFINE_integer('img_shape', [436, 1024, 3],
                     'Image shape: width, height, channels')

flags.DEFINE_integer('flow_shape', [436, 1024, 2],
                     'Image shape: width, height, 2')

flags.DEFINE_integer('record_bytes', 3571724,
                           'Flow record bytes reader for Sintel')

# HYPERPARAMETER

flags.DEFINE_integer('batchsize', 2, 'Batch size.')

flags.DEFINE_integer('testsize', 1041,
                     'Number of training steps.')

flags.DEFINE_integer('drop_rate', 0.5, 'Dropout change')

flags.DEFINE_boolean('batch_normalization', False, 'Batch on/off')

flags.DEFINE_boolean('weights_reg', None,
                     'Weights regularizer')

flags.DEFINE_boolean('is_training', False, 'Batch on/off')
# TESTING

flags.DEFINE_integer('img_summary_num', 1, 'Number of images in summary')

flags.DEFINE_string('master', '',
                    'BNS name of the TensorFlow master to use.')

flags.DEFINE_integer('eval_interval_secs', 300,
                     'How many seconds between executions of the eval loop.')

# bilateral solver

flags.DEFINE_string('dataset', '/sintel_final',
                    'Dataset to test')

# sintel clean (Improvenemt ~0.1 EPE)
flags.DEFINE_string('grid_params', { 'sigma_luma' : 5,'sigma_chroma': 2, 'sigma_spatial': 11},
                    'grid_params')

flags.DEFINE_string('bs_params', {'lam': 332, 'A_diag_min': 1e-5, 'cg_tol': 1e-5, 'cg_maxiter': 43},
                    'bs_params')

# sintel final (Improvenemt ~0.05 EPE)
"""
flags.DEFINE_string('grid_params', { 'sigma_luma' : 3,'sigma_chroma': 2, 'sigma_spatial': 10},
                    'grid_params')

flags.DEFINE_string('bs_params', {'lam': 314, 'A_diag_min': 1e-5, 'cg_tol': 1e-5, 'cg_maxiter': 29},
                    'bs_params')
"""

flags.DEFINE_integer('flow_int', 1,
                     'Variable for img saving.')

def aee_f(gt, calc_flows):
    "average end point error"
    square = tf.square(gt - calc_flows)
    square = tf.squeeze(square)
    x , y = tf.split(square, num_or_size_splits=2, axis=2)
    sqr = tf.sqrt(tf.add(x, y))
    aee = tf.metrics.mean(sqr)
    return aee

def var_mean(flow_to_mean):
    """ Pyfunc wrapper for the confidence / mean calculation"""

    def _var_mean(flow_to_mean):
        """ confidence / mean calculation"""
        flow_to_mean = np.array(flow_to_mean)
        x = flow_to_mean[:,:,:,0]
        y = flow_to_mean[:,:,:,1]
        var_x = np.var(x, 0)
        var_y = np.var(y, 0)

        # variance - confidence, this could be done differently e.g. exp(-1*...)
        conf_x =  1 - var_x/np.amax(var_x)
        conf_y =  1 - var_y/np.amax(var_y)

        # mean flow
        mean_flow = np.zeros(list(FLAGS.d_shape_flow), np.float32)
        mean_flow[:,:,0] = np.mean(x, 0)
        mean_flow[:,:,1] = np.mean(y, 0)

        # make confidence image (all color chanels the same)
        conf_img = np.zeros(list(FLAGS.d_shape_img), np.float32)
        conf_img[:,:,0] = (conf_x + conf_y)/2.0
        conf_img[:,:,1] = (conf_x + conf_y)/2.0
        conf_img[:,:,2] = (conf_x + conf_y)/2.0
        return [mean_flow, conf_x, conf_y, conf_img]

    mean_flow, conf_x, conf_y, conf_img  = tf.py_func( _var_mean, [flow_to_mean],
                                        [tf.float32, tf.float32, tf.float32, tf.float32], name='mean_flow')[:]

    mean_flow = tf.squeeze(tf.stack(mean_flow))
    conf_x = tf.squeeze(tf.stack(conf_x))
    conf_y = tf.squeeze(tf.stack(conf_y))
    conf_img = tf.squeeze(tf.stack(conf_img))

    mean_flow.set_shape(list(FLAGS.d_shape_flow))
    conf_x.set_shape(list(FLAGS.d_shape_img[:2])+ [1])
    conf_y.set_shape(list(FLAGS.d_shape_img[:2])+ [1])
    conf_img.set_shape(list(FLAGS.d_shape_img))

    return mean_flow, conf_x, conf_y, conf_img

def main(_):

    """Evaluate FlowNet for Sintel test set"""

    with tf.Graph().as_default():

        # just get one triplet at a time
        var_num = 1
        img_0, img_1, flow = flownet_tools.get_data_sintel(FLAGS.datadir, False, var_num)

        # reize from [436, 1024] to [448, 1024] since 436/2**3 not even -> stride problems
        # this is a change of height ~3% -> scale of flow in vertical direction needed
        img_0_rs = tf.squeeze(tf.image.resize_images(img_0, FLAGS.d_shape_img[:2]))
        img_1_rs = tf.squeeze(tf.image.resize_images(img_1, FLAGS.d_shape_img[:2]))
        flow_rs = tf.squeeze(tf.image.resize_images(flow, FLAGS.d_shape_img[:2]))

        ratio_h = tf.cast(FLAGS.d_shape_img[0], tf.float32) / FLAGS.img_shape[0]
        ratio_w = tf.cast(FLAGS.d_shape_img[1], tf.float32) / FLAGS.img_shape[1]
        flow_rs = tf.squeeze(tf.stack([tf.split(flow_rs, 2, axis=-1)[0]*ratio_w,
                                tf.split(flow_rs, 2, axis=-1)[1]*ratio_h], -1))

        # stack for simple multiple inference
        imgs_0_rs = tf.squeeze(tf.stack([img_0_rs for i in range(FLAGS.batchsize)]))
        imgs_1_rs = tf.squeeze(tf.stack([img_1_rs for i in range(FLAGS.batchsize)]))
        flows_rs = tf.squeeze(tf.stack([flow_rs for i in range(FLAGS.batchsize)]))

        # img summary after loading
        flownet.image_summary(imgs_0_rs, imgs_1_rs, "A_input", flows_rs)

        calc_flows = model(imgs_0_rs, imgs_1_rs, flows_rs)

        if FLAGS.dropout and FLAGS.is_training:
            flow_split=tf.split(calc_flows,FLAGS.batchsize, axis=0)
            # calc mean / variance and images for that
            aee_mean = np.zeros(FLAGS.batchsize)
            mean_di = {}
            for i in range(1, FLAGS.batchsize):
                calc_flows = tf.squeeze(tf.stack([flow_split[:i+1]]))
                mean_flow, conf_x, conf_y, conf_img  = var_mean(calc_flows)
                mean_di[i] = aee_f(flow_rs, mean_flow)
            # start bilateral solver
            img_0 = tf.squeeze(tf.stack(img_0))
            flow_s = tf.squeeze(tf.stack(flow_rs))
            solved_flow = flownet.bil_solv_var(img_0_rs, mean_flow, conf_x, conf_y, flow_s)
            # calc aee for solver
            aee_bs = aee_f(flow_rs, solved_flow)

            # MC dropout / Mean for different numbers of samples
            metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
                      "AEE_2": slim.metrics.streaming_mean(mean_di[1]),
                      "AEE_10": slim.metrics.streaming_mean(mean_di[9]),
                      "AEE_25": slim.metrics.streaming_mean(mean_di[24]),
                      "AEE_40": slim.metrics.streaming_mean(mean_di[39]),
                      "AEE_bs": slim.metrics.streaming_mean(aee_bs),
            })
        else:
            # this is for weight scaling - is_training=False
            # or when dropout is off
            calc_flow = tf.squeeze(tf.split(calc_flows, FLAGS.batchsize, axis=0)[0])
            aae = aee_f(flow_rs, calc_flow)
            metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
                      "AEE": slim.metrics.streaming_mean(aae),
            })

        # write summary
        for name, value in metrics_to_values.iteritems():
            		tf.summary.scalar(name, value)

        # summary images
        flownet.image_summary(None, None, "FlowNetS", calc_flows)
        if FLAGS.dropout and FLAGS.is_training:
            solved_flows = tf.squeeze(tf.stack([solved_flow for i in range(FLAGS.batchsize)]))
            mean_flows = tf.squeeze(tf.stack([mean_flow for i in range(FLAGS.batchsize)]))
            conf_imgs = tf.squeeze(tf.stack([conf_img for i in range(FLAGS.batchsize)]))
            flownet.image_summary(None, None, "FlowNetS BS", solved_flows)
            flownet.image_summary(None, None, "FlowNetS Mean", mean_flows)
            flownet.image_summary(None, None, "Confidence", conf_imgs)

        # Run the actual evaluation loop.
        num_batches = math.ceil(FLAGS.testsize) - 1
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True

        slim.evaluation.evaluation_loop(
        	master=FLAGS.master,
            checkpoint_dir=FLAGS.logdir + '/train',
            logdir=FLAGS.logdir + '/eval_sintel_clean_scale',
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
      default='log_drop_rot',
      help='Directory where to write event logs and checkpoints'
    )
    parser.add_argument(
        '--dropout',
        type=str,
        default='false',
        help='Trun dropout on/off'
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
    args = parser.parse_known_args()[0]
    FLAGS.datadir = os.path.join(dir_path,  parser.parse_args().datadir)
    FLAGS.logdir = os.path.join(dir_path, parser.parse_args().logdir)
    FLAGS.imgsummary = args.imgsummary

    drop = args.dropout
    if drop.lower() in ('yes', 'true'):
        FLAGS.dropout = True
        print("Dropout on")
    elif drop.lower() in ('no', 'false'):
        FLAGS.dropout = False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

    # turn weight decay on/off
    if args.weights_reg != 0:
        print("Weight decay with: " + str(args.weights_reg))
        FLAGS.weights_reg = slim.l1_regularizer(args.weights_reg)
    else:
    	FLAGS.weights_reg = None

    print("Using architecture: " + model.__name__)
    tf.app.run()
