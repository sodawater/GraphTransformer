from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.keras import backend as K
import math

def index_matrix_to_pairs(index_matrix):
  # [[3,1,2], [2,3,1]] -> [[[0, 3], [1, 1], [2, 2]],
  #                        [[0, 2], [1, 3], [2, 1]]]
  replicated_first_indices = tf.range(tf.shape(index_matrix)[0])
  rank = len(index_matrix.get_shape())
  if rank == 2:
    replicated_first_indices = tf.tile(
        tf.expand_dims(replicated_first_indices, dim=1),
        [1, tf.shape(index_matrix)[1]])
  return tf.stack([replicated_first_indices, index_matrix], axis=rank)

def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * tf.reduce_sum(1 + (recog_logvar - prior_logvar)
                               - tf.div(tf.pow(prior_mu - recog_mu, 2), tf.exp(prior_logvar))
                               - tf.div(tf.exp(recog_logvar), tf.exp(prior_logvar)), reduction_indices=1)
    return kld

def gelu(input_tensor):
  cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf

def leaky_relu(features, alpha=0.2, name=None):
  """Compute the Leaky ReLU activation function.
  "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
  AL Maas, AY Hannun, AY Ng - Proc. ICML, 2013
  https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
  Args:
    features: A `Tensor` representing preactivation values. Must be one of
      the following types: `float16`, `float32`, `float64`, `int32`, `int64`.
    alpha: Slope of the activation function at x < 0.
    name: A name for the operation (optional).
  Returns:
    The activation value.
  """
  with ops.name_scope(name, "LeakyRelu", [features, alpha]) as name:
    features = ops.convert_to_tensor(features, name="features")
    if features.dtype.is_integer:
      features = math_ops.to_float(features)
    alpha = ops.convert_to_tensor(alpha, dtype=features.dtype, name="alpha")
    return math_ops.maximum(alpha * features, features, name=name)

def selu(x):
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946
  return scale * K.elu(x, alpha)

def norm_log_liklihood(x, mu, logvar):
    return -0.5*tf.reduce_sum(tf.log(2*np.pi) + logvar + tf.div(tf.pow((x-mu), 2), tf.exp(logvar)), reduction_indices=1)



def reverse(input_, seq_lengths, seq_dim, batch_dim):
    if seq_lengths is not None:
        return array_ops.reverse_sequence(
            input=input_, seq_lengths=seq_lengths,
            seq_dim=seq_dim, batch_dim=batch_dim)
    else:
        return array_ops.reverse(input_, axis=[seq_dim])

def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def multihead_attention(queries,
                        keys,
                        query_length,
                        key_length,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        pointer=False,
                        using_mask=False,
                        no_tile=False,
                        mymasks=None,
                        scope="multihead_attention",
                        reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):

        if num_units is None:
            num_units = queries.get_shape().as_list[-1]



        Q = tf.layers.dense(queries, num_units, activation=None, use_bias=False, name="q")  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None, use_bias=False, name="k")  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=None, use_bias=False, name="v")  # (N, T_k, C)


        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication

        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        if key_length != None:
            key_masks = tf.sequence_mask(key_length, tf.shape(keys)[1], dtype=tf.float32)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)


        if pointer == True:
            outputs = tf.nn.softmax(outputs)

            return outputs

        if using_mask:
            if not no_tile:
                mymask = tf.tile(mymasks, [num_heads, 1, 1])
            else:
                mymask = mymasks
            outputs = tf.where(tf.equal(mymask, 0), paddings, outputs)



        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        query_masks = tf.sequence_mask(query_length, tf.shape(queries)[1], dtype=tf.float32)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)

        outputs *= query_masks


        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
        outputs = tf.layers.dense(tf.concat(tf.split(outputs, num_heads, axis=0), axis=2), num_units, activation=None,
                                  use_bias=False, name="concat")  # (N, T_q, C)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        return outputs

def positional_encoding(length,
                        inputs,
                        num_units,
                        zero_pad=False,
                        scope="positional_encoding",
                        reuse=None):

    E = num_units # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)
        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos * np.exp(-np.log(10000.0) * ((i * 1.0 - i % 2)/2) / (E * 1.0 /2)) for i in range(E)]
            for pos in range(length)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)


        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        if zero_pad:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)








def feedforward(inputs,
                num_units=[2048, 512],
                scope="feedforward",
                is_training=False,
                dropout_rate=0,
                reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):

        outputs = tf.layers.dense(inputs, num_units[0], activation=gelu)

        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        outputs = tf.layers.dense(outputs, num_units[1])

        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
    return outputs