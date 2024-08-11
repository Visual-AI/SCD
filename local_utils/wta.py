import tensorflow.compat.v2 as tf
import numpy as np
tf.enable_v2_behavior()

def get_a_fake_feature(batch_size, feat_dim):
  return tf.random.normal([batch_size, feat_dim], mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)

def get_structured_hash_idx(embed_dim, hash_code_dim, hash_win_len):
  embed_idx_mat = tf.tile(tf.expand_dims(tf.range(embed_dim), axis=0), multiples=[hash_code_dim, 1])
  # Random shuffle each row, obtain a matrix with shape [hash_code_dim, hash_win_len]
  structured_idx_mat = tf.map_fn(tf.random.shuffle, embed_idx_mat)[:, :hash_win_len]
  return structured_idx_mat

def hash_transform(embed_feature, hash_code_dim, hash_win_len):
  if len(embed_feature.get_shape().as_list()) != 2:
    raise ValueError('Unexpected embed_feature shape, whereas a 2 dimension tensor is required')
  batch_size = tf.shape(embed_feature)[0]
  embed_dim = tf.shape(embed_feature)[1]
  structured_index_mat = get_structured_hash_idx(embed_dim, hash_code_dim, hash_win_len)
  structured_hash_code = tf.gather(params=embed_feature, indices=tf.reshape(structured_index_mat, (-1,)), axis=-1)
  structured_hash_code = tf.reshape(structured_hash_code, shape=(batch_size, hash_code_dim, hash_win_len))
  # only keep the most significant feature in a local comparison window
  structured_hash_code = tf.math.argmax(structured_hash_code, axis=-1)
  return structured_hash_code  