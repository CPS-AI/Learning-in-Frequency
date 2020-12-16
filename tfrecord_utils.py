from functools import partial

import tensorflow as tf
import numpy as np

def decode_feature(feature, wide, feature_dim):
    feature = tf.expand_dims(feature, 0)
    feature = tf.reshape(feature, shape=(wide, feature_dim))
    return feature

def read_tfrecord(example, wide, feature_dim, out_dim):
    example_description = {
        'class': tf.io.FixedLenFeature([out_dim], tf.float32),
        'example': tf.io.FixedLenFeature([wide*feature_dim], tf.float32),
    }
    example = tf.io.parse_single_example(example, example_description)
    feature = decode_feature(example['example'], wide, feature_dim)
    label = example['class']
    return feature, label

def load_dataset(tfrec_path, wide, feature_dim, out_dim):
    dataset = tf.data.TFRecordDataset(tfrec_path)
    dataset = dataset.map(
        partial(read_tfrecord, 
            wide = wide, 
            feature_dim = feature_dim,
            out_dim=out_dim
            ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    return dataset

def get_dataset(tfrec_path, batch_size, wide, feature_dim, out_dim, shuffle_sample=True):
    dataset = load_dataset(tfrec_path, wide, feature_dim, out_dim)
    if shuffle_sample:
        dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset

# def read_and_decode(tfrec_path, wide, feature_dim, out_dim):
#     dataset = tf.data.TFRecordDataset(tfrec_path)

#     image_feature_description = {
#         'class': tf.FixedLenFeature([out_dim], tf.float32),
#         'example': tf.FixedLenFeature([wide*feature_dim], tf.float32),
#     }

#     features = tf.parse_single_example(dataset,
#                                       image_feature_description)
#     return features['example'], features['class']

# def input_pipeline(tfrec_path, batch_size, wide, feature_dim, out_dim, shuffle_sample=True, num_epochs=None):
#     example, label = read_and_decode(tfrec_path, wide, feature_dim, out_dim)
#     example = tf.expand_dims(example, 0)
#     example = tf.reshape(example, shape=(wide, feature_dim))
    
#     if shuffle_sample:



#     min_after_dequeue = 1000  # int(0.4*len(csvFileList)) #1000
#     capacity = min_after_dequeue + 3 * batch_size
#     if shuffle_sample:
#         example_batch, label_batch = tf.train.shuffle_batch(
#             [example, label], batch_size=batch_size, num_threads=16, capacity=capacity,
#             min_after_dequeue=min_after_dequeue)
#     else:
#         example_batch, label_batch = tf.train.batch(
#             [example, label], batch_size=batch_size, num_threads=16)

#     return example_batch, label_batch

