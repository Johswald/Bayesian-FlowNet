# FlowNet in Tensorflow
# InputData
# ==============================================================================

import collections
import glob

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import flags

import readFlowFile

FLAGS = flags.FLAGS

Datasets = collections.namedtuple('Datasets', ['train', 'test'])

class DataSet(object):

  def __init__(self,
               images_0,
               images_1,
               flows,
               dtype=dtypes.float32,
               shuffle=True):

    """Construct a DataSet """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
    # check input data for consistency
    # check length
    assert len(images_0) == len(images_1) == len(flows), (
    'images_0 length: %s images_1 length: %s flows length: %s' % 
    (len(images_0), len(images_1), len(flows)))

    # get random sample and test shapes
    random = np.random.randint(0, len(images_0))
    image_0, image_1 = load_images([images_0[random]], [images_1[random]])
    flow = load_flows([flows[random]])

    #check shape (of first image, flow)
    assert image_0[0].shape == image_1[0].shape and image_1[0].shape[:2] == flow[0].shape[:2], (
    'image_0[0].shape: %s image_1[0].shape: %s flows[1].shape[:2]: %s' % 
    (image_0[0].shape[:2], image_1[0].shape[:2], flow[0].shape[:2]))

    # if true shuffle list
    if shuffle:
      p = np.random.permutation(len(images_0))
      images_0 = [images_0[i] for i in p]
      images_1 = [images_1[i] for i in p]
      flows = [flows[i] for i in p]

    self._num_examples = len(images_0)
    self._images_0 = images_0
    self._images_1 = images_1
    self._flows = flows
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images_0, self._images_1

  @property
  def flows(self):
    return self._flows

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next batch_size from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      self._epochs_completed += 1
      self._index_in_epoch = 0
      start = self._index_in_epoch
      self._index_in_epoch += batch_size
      p = np.random.permutation(len(self._images_0))
      self._images_0 = [self._images_0[i] for i in p]
      self._images_1 = [self._images_1[i] for i in p]
      self._flows = [self._flows[i] for i in p]
    end = self._index_in_epoch
    return load_images(self._images_0[start:end], self._images_1[start:end]), load_flows(self._flows[start:end])

def load_images(img_files_0_LIST, img_files_1_LIST):
  """Load Images 0 and 1 with cv2 from given lists
  Problem: Too slow? Use rawreader from tensorflow?
  """
  images_0 = []
  images_1 = []
  for image_0, image_1 in zip(img_files_0_LIST, img_files_1_LIST):
    img_0 = cv2.imread(image_0)
    img_1 = cv2.imread(image_1)
    img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB) 
    images_0.append(img_0)
    images_1.append(img_1)  
  return images_0, images_1

def load_flows(flo_files_LIST):
  """Load ground trouth flows from given lists
  """
  flows = []
  for flow in flo_files_LIST:
    flows.append(readFlowFile.read(flow))
  return flows

def read_data_lists(data_dir, split_list):
  """Reads data from fiven dir and creates dataset objects for training"""
  # get list of all files
  from operator import itemgetter as ig

  train_images_0 = sorted(glob.glob(data_dir + "/*1.ppm"))
  train_images_1 = sorted(glob.glob(data_dir + "/*2.ppm"))
  train_flows = sorted(glob.glob(data_dir + "/*.flo"))

  assert train_images_0 != train_images_1 != train_flows, (
    'lengths of images / flows are not the same: %s, %s, and %s' %
    (len(train_images_0), len(train_images_1), len(train_flows)))

  if len(train_images_0) == 22872:
    split_text = open(split_list, 'rb')
    split_list = []
    for line in split_text:
      split_list.append(line)
    split_text.close()
    test = []
    train = []
    e = 0
    r = 0
    for line, i in zip(split_list, range(len(split_list))):
      if "1" in line:
        train.append(i)
      elif "2" in line:
        test.append(i)
      else:
        print("Split list errror")
        exit()
    if len(ig(*train)(train_images_0)) + len(ig(*test)(train_images_0)) != len(train_images_0):
      print("Split list errror")
      exit()
    
    train = DataSet(ig(*train)(train_images_0), ig(*train)(train_images_1), ig(*train)(train_flows))
    test = DataSet(ig(*test)(train_images_0), ig(*test)(train_images_1), ig(*test)(train_flows))
    return Datasets(train=train, test=test)
  else:
    train = DataSet(list(train_images_0), list(train_images_1), list(train_flows))
    test = DataSet(list(train_images_0), list(train_images_1), list(train_flows))
    return Datasets(train=train, test=test)
  


