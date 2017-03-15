#!/usr/bin/env python
# -----------------------------------
# separate the training and testing data and convert images to tfrecords
# Author: Tao Chen
# Date: 2016.10.16
# -----------------------------------

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import collections
import tensorflow as tf
import glob
import yaml
import time
import random
import math
import numpy as np
from PIL import Image
Dataset = collections.namedtuple('Dataset', ['left_images', 'right_images', 'labels'])
FLAGS = None


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(dataset, name):
  """Converts a dataset to tfrecords."""
  images_left = dataset.left_images
  images_right = dataset.right_images
  labels = dataset.labels
  if images_left.shape[0] != labels.shape[0] or images_left.shape[0] != images_right.shape[0]\
          or images_right.shape[0] != labels.shape[0]:
      raise ValueError('Images_left size %d, Images_right size %d, and label size %d do not match.' %
                       (images_left.shape[0], images_right.shape[0], labels.shape[0]))

  if images_left.shape != images_right.shape:
      raise ValueError('Images_left shape {0}, Images_right shape {1} do not match.'.format(images_left.shape,
                                                                                            images_right.shape))
  num_examples = images_left.shape[0]
  rows = images_left.shape[1]
  cols = images_left.shape[2]
  depth = images_left.shape[3]

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  # print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_left_raw = images_left[index].tostring()
    image_right_raw = images_right[index].tostring()
    label = labels[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _bytes_feature(label),
        'image_left': _bytes_feature(image_left_raw),
        'image_right': _bytes_feature(image_right_raw)}))
    writer.write(example.SerializeToString())
  writer.close()

def convert_data_sets(directory, train_ratio = 0.8, num_partition = 10):
    left_images_dir = os.path.join(directory, 'left')
    right_images_dir = os.path.join(directory, 'right')
    labels_dir = os.path.join(directory, 'ball_pos')
    left_images_list = glob.glob('%s/*.jpg'%(left_images_dir))
    right_images_list = glob.glob('%s/*.jpg' % (right_images_dir))
    labels_list = glob.glob('%s/*.yaml' % (labels_dir))

    if len(left_images_list) != len(right_images_list) or len(labels_list) != len(right_images_list) \
            or len(left_images_list) != len(labels_list):
        raise ValueError('Images_left size %d, Images_right size %d, and label size %d do not match.' %
                         (len(left_images_list), len(right_images_list), len(labels_list)))

    random.shuffle(left_images_list)
    camera_pos = np.array([0.7, 3.8, 1.8])
    num_examples = len(left_images_list)
    start = 0
    batch = math.ceil(num_examples / num_partition)
    print('batch_size:',batch)
    for epoch in range(num_partition):
        end = int(min(start + batch, num_examples))
        left_images = []
        right_images = []
        labels = []
        # print('start = ', start)
        # print('end = ', end)
        for left_image_path in left_images_list[start:end]:
            left_image_filename = left_image_path.split('/')[-1]
            right_file = left_image_filename.replace('L', 'R')
            label_file = left_image_filename.replace('L', 'ball_pos')
            label_file = label_file.replace('jpg', 'yaml')
            right_image_path = left_image_path.replace(os.path.join('left', left_image_filename), os.path.join('right', right_file))
            label_path = left_image_path.replace(os.path.join('left', left_image_filename), os.path.join('ball_pos', label_file))
            if right_image_path not in right_images_list or label_path not in labels_list:
                continue
            with open(label_path, 'r') as stream:
                label = yaml.load(stream)['ball_pos']
            left_image = np.asarray(Image.open(left_image_path))
            right_image = np.asarray(Image.open(right_image_path))
            # # For debug
            # Image.fromarray(left_image).show()
            # Image.fromarray(right_image).show()

            left_images.append(left_image)
            right_images.append(right_image)
            labels.append(label)
        left_images_asarray = np.array(left_images)
        right_images_asarray = np.array(right_images)
        labels_asarray = np.array(labels)
        labels_asarray -= camera_pos

        num_train = int((end - start) * train_ratio)
        train_left_images = left_images_asarray[:num_train]
        train_right_images = right_images_asarray[:num_train]
        train_labels = labels_asarray[:num_train]

        test_left_images = left_images_asarray[num_train:]
        test_right_images = right_images_asarray[num_train:]
        test_labels = labels_asarray[num_train:]

        train_dataset = Dataset(left_images=train_left_images, right_images=train_right_images, labels=train_labels)
        test_dataset = Dataset(left_images=test_left_images, right_images=test_right_images, labels=test_labels)
        convert_to(train_dataset, 'train-%.5d' % epoch)
        convert_to(test_dataset, 'test-%.5d' % epoch)
        start = end


def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return float(total_size) / 1024.0 / 1024.0 / 1024.0

def main(unused_argv):
    start_time = time.time()
    convert_data_sets(FLAGS.directory, train_ratio = 0.8, num_partition=10)
    end_time = time.time()
    print('elapsed time:', end_time - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        default='data',
        help='Directory to download data files and write the converted result'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
