"""
FlowNetS training module
"""

import argparse
import os
from os.path import dirname

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import flags
from tensorflow.python.training import saver as tf_saver
import flownet
import flownet_tools
import architectures

dir_path = dirname(os.path.realpath(__file__))

# Basic model parameters as external flags.
FLAGS = flags.FLAGS

# ARCHITECTURE

model = architectures.flownet_s

# DATA

flags.DEFINE_integer('d_shape_img', [384, 512, 3],
                           'Data shape: width, height, channels')

flags.DEFINE_integer('d_shape_flow', [384, 512, 2],
                           'Data shape: width, height, channels')

flags.DEFINE_integer('img_shape', [384, 512, 3],
                     'Image shape: width, height, channels')

flags.DEFINE_integer('flow_shape', [384, 512, 2],
                     'Image shape: width, height, 2')

flags.DEFINE_integer('record_bytes', 1572876,
                           'Flow record bytes reader for FlyingChairs')

# HYPERPARAMETER

flags.DEFINE_integer('batchsize', 8, 'Batch size.')

flags.DEFINE_integer('max_steps', 800000,
                     'Number of training steps.')

flags.DEFINE_integer('boundaries', [i*100000 for i in range(3, FLAGS.max_steps/100000)],
					'boundaries for learning rate')

flags.DEFINE_integer('values', [1e-4/(2**i) for i in range(0, FLAGS.max_steps/100000-2)],
					'learning rate values')

flags.DEFINE_integer('learning_rate', 1e-4, 'learning rate values')

flags.DEFINE_integer('drop_rate', 0.5, 'Dropout change')

flags.DEFINE_boolean('batch_normalization', False, 'Batch on/off')

flags.DEFINE_boolean('is_training', True, 'Batch on/off')
# TRAINING

flags.DEFINE_integer('img_summary_num', 1, 'Number of images in summary')

flags.DEFINE_integer('max_checkpoints', 5, 'Maximum number of recent checkpoints to keep.')

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

def apply_augmentation(imgs_0, imgs_1, flows):
    # apply data augmenation to data batch

    if FLAGS.augmentation:
        with tf.name_scope('Augmentation'):

            # chromatic tranformation of images
            imgs_0, imgs_1 = flownet.fast_chromatic_augm(imgs_0, imgs_1)

            #rotation / scaling / cropping (very important for flow)
            imgs_0, imgs_1, flows = flownet.rotation_crop_trans(imgs_0, imgs_1, flows)
            flownet.image_summary(imgs_0, imgs_1, "B_after_augm", flows)

    return imgs_0, imgs_1, flows


def main(_):
    """Train FlowNet for a FLAGS.max_steps."""

    with tf.Graph().as_default():

        imgs_0, imgs_1, flows = flownet_tools.get_data(FLAGS.datadir, True)
        # img summary after loading
        flownet.image_summary(imgs_0, imgs_1, "A_input", flows)

        # apply augmentation
        imgs_0, imgs_1, flows = apply_augmentation(imgs_0, imgs_1, flows)

        # model
        calc_flows = model(imgs_0, imgs_1, flows)

        # img summary of result
        flownet.image_summary(None, None, "E_result", calc_flows)

        global_step = slim.get_or_create_global_step()
        train_op = flownet.create_train_op(global_step)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datadir',
        type=str,
        default='data/flying/train/',
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--logdir',
        type=str,
        default='with_data_aug',
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
        type=str,
        default="true",
        help='Make image summary'
    )
    parser.add_argument(
        '--augmentation',
        type=str,
        default="true",
        help='Make data augmentation'
    )
    parser.add_argument(
        '--weights_reg',
        type=float,
        default=0,
        help='weights regularizer'
    )
    args = parser.parse_known_args()[0]
    FLAGS.datadir = os.path.join(dir_path,  args.datadir)
    FLAGS.logdir = os.path.join(dir_path, args.logdir)

    # get boolean if data augmentation is wanted
    aug = args.augmentation
    if aug.lower() in ('yes', 'true'):
        FLAGS.augmentation = True
        print("Data augmentation on")
    elif aug.lower() in ('no', 'false'):
        FLAGS.augmentation = False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

    drop = args.dropout
    if drop.lower() in ('yes', 'true'):
        FLAGS.dropout = True
        print("Dropout on")
    elif drop.lower() in ('no', 'false'):
        FLAGS.dropout = False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

    # get boolean if img summary is wanted
    img_ = args.imgsummary
    if img_.lower() in ('yes', 'true'):
        FLAGS.imgsummary = True
    elif img_.lower() in ('no', 'false'):
        FLAGS.imgsummary = False
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
