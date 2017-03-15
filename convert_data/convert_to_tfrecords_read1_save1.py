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
import random
import numpy as np
import time
from PIL import Image
Dataset = collections.namedtuple('Dataset', ['left_images_path', 'right_images_path', 'labels_path'])
FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_data_sets(directory, train_ratio = 0.8):
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
    num_train = int(len(left_images_list) * train_ratio)
    camera_pos = np.array([0.7, 3.8, 1.8])

    train_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.directory, 'train.tfrecords'))
    test_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.directory, 'test.tfrecords'))
    for index, left_image_path in enumerate(left_images_list):
        left_image_filename = left_image_path.split('/')[-1]
        right_file = left_image_filename.replace('L', 'R')
        label_file = left_image_filename.replace('L', 'ball_pos')
        label_file = label_file.replace('jpg', 'yaml')
        right_image_path = left_image_path.replace(os.path.join('left', left_image_filename), os.path.join('right', right_file))
        label_path = left_image_path.replace(os.path.join('left', left_image_filename), os.path.join('ball_pos', label_file))
        if right_image_path not in right_images_list or label_path not in labels_list:
            continue

        with open(label_path, 'r') as stream:
            label = np.array(yaml.load(stream)['ball_pos']) - camera_pos
        image_left = np.asarray(Image.open(left_image_path))
        image_right = np.asarray(Image.open(right_image_path))
        if image_left.shape != image_right.shape:
            raise ValueError('Image_left shape {0}, Image_right shape {1} do not match.'.format(image_left.shape,
                                                                                                image_right.shape))
        rows = image_left.shape[0]
        cols = image_left.shape[1]
        depth = image_left.shape[2]

        # print('Writing', filename)
        image_left_raw = image_left.tostring()
        image_right_raw = image_right.tostring()
        label = label.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _bytes_feature(label),
            'image_left': _bytes_feature(image_left_raw),
            'image_right': _bytes_feature(image_right_raw)}))
        if index < num_train:
            train_writer.write(example.SerializeToString())
        else:
            test_writer.write(example.SerializeToString())
    train_writer.close()
    test_writer.close()


def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return float(total_size) / 1024.0 / 1024.0 / 1024.0

def main(unused_argv):
    start_time = time.time()
    convert_data_sets(FLAGS.directory, train_ratio=0.8)

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
