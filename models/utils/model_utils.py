import operator
from functools import reduce

import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_estimator

from . import dropouts
from .optimizers import get_optimizer
from ...training import tpu


class ModeKeys(object):
  TRAIN = tf.estimator.ModeKeys.TRAIN
  EVAL = tf.estimator.ModeKeys.EVAL
  TEST = "test"
  PREDICT = tf.estimator.ModeKeys.PREDICT
  ATTACK = "attack"


def collect_vars(fn):
  """Collect all new variables created within `fn`.

  Args:
    fn: a function that takes no arguments and creates trainable tf.Variable
      objects.

  Returns:
    outputs: the outputs of `fn()`.
    new_vars: a list of the newly created variables.
  """
  previous_vars = set(tf.trainable_variables())
  outputs = fn()
  current_vars = set(tf.trainable_variables())
  new_vars = current_vars.difference(previous_vars)
  return outputs, list(new_vars)


def dense(x, units, hparams, is_training, dropout=True):
  with tf.variable_scope(None, default_name="dense") as scope:
    w = tf.get_variable("kernel", shape=[x.shape[1], units], dtype=tf.float32)
    b = tf.get_variable(
        "bias",
        shape=[units],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())
    if dropout and hparams.dropout_type is not None and is_training:
      w = dropouts.get_dropout(hparams.dropout_type)(w, hparams, is_training)

    w = tf.identity(w, name="post_dropout")
    y = tf.matmul(x, w) + b
    return y


def conv(x,
         filter_size,
         out_filters,
         hparams,
         strides=[1, 1, 1, 1],
         padding="SAME",
         is_training=False,
         activation=None,
         dropout=True,
         name=None,
         schit_layer=False):
  """Convolution."""
  with tf.variable_scope(name, default_name="conv2d"):
    if hparams.data_format == "channels_last":
      in_filters = x.shape[-1]
    else:
      in_filters = x.shape[1]

    kernel = tf.get_variable(
        'DW', [filter_size, filter_size, in_filters, out_filters], tf.float32)
    use_dropout = hparams.dropout_type is not None and dropout

    # schit layer
    if schit_layer:
      scale = tf.get_variable(
          'scale',
          kernel.shape[-1],
          tf.float32,
          initializer=tf.zeros_initializer())
      kernel = hparams.lipschitz_constant * tf.nn.sigmoid(
          scale) * kernel / tf.norm(
              tf.reshape(kernel, shape=[-1, kernel.shape[-1]]), axis=0)

    if use_dropout:
      dropout_fn = dropouts.get_dropout(hparams.dropout_type)

      if hparams.dropout_type == "targeted_ard":
        kernel = dropout_fn(kernel, x, hparams, is_training)
      else:
        kernel = dropout_fn(kernel, hparams, is_training)

      # special case for variational
      if "variational" in hparams.dropout_type:
        kernel, log_alpha = kernel[0], kernel[1]
        if is_training:
          conved_mu = tf.nn.conv2d(x, kernel, strides=strides, padding=padding)
          conved_si = tf.sqrt(
              tf.nn.conv2d(
                  tf.square(x),
                  tf.exp(log_alpha) * tf.square(kernel),
                  strides=strides,
                  padding=padding) + 1e-8)
          conved = conved_mu + tf.random_normal(
              tf.shape(conved_mu)) * conved_si

          conved = tf.identity(conved, name="post_dropout")
          return conved

    data_format = "NHWC" if hparams.data_format == "channels_last" else "NCHW"
    conv = tf.nn.conv2d(
        x, kernel, strides, padding=padding, data_format=data_format)

    if activation:
      conv = activation(conv)

    conv = tf.identity(conv, name="post_dropout")
    return conv


def weight_decay_and_noise(loss, hparams, learning_rate, var_list=None):
  """Apply weight decay and weight noise."""

  weight_decay_loss = weight_decay(hparams)
  tf.summary.scalar("losses/weight_decay", weight_decay_loss)
  weight_noise_ops = weight_noise(hparams, learning_rate)
  with tf.control_dependencies(weight_noise_ops):
    loss = tf.identity(loss)

  loss += weight_decay_loss
  return loss


def weight_noise(hparams, learning_rate):
  """Apply weight noise to vars in var_list."""
  if not hparams.weight_noise_rate:
    return [tf.no_op()]

  tf.logging.info("Applying weight noise scaled by learning rate, "
                  "noise_rate: %0.5f", hparams.weight_noise_rate)
  noise_ops = []

  noise_vars = [v for v in tf.trainable_variables() if "/body/" in v.name]
  for v in var_list:
    with tf.device(v._ref().device):  # pylint: disable=protected-access
      scale = hparams.weight_noise_rate * learning_rate * 0.001
      tf.summary.scalar("weight_noise_scale", scale)
      noise = tf.truncated_normal(v.shape) * scale
      noise_op = v.assign_add(noise)
      noise_ops.append(noise_op)
  return noise_ops


def weight_decay(hparams):
  """Apply weight decay to vars in var_list."""
  if not hparams.weight_decay_rate:
    return 0.

  only_features = hparams.weight_decay_only_features
  var_list = [v for v in tf.trainable_variables()]
  weight_decays = []
  for v in var_list:
    # Weight decay.
    is_feature = any(n in v.name for n in hparams.weight_decay_weight_names)
    if (not only_features) or is_feature:
      if hparams.initializer == "hadamard_unscaled":
        v_loss = tf.reduce_sum((tf.abs(v) - 1)**2) / 2
      else:
        v_loss = tf.nn.l2_loss(v)
      weight_decays.append(v_loss)

  return tf.reduce_sum(weight_decays, axis=0) * hparams.weight_decay_rate


def axis_aligned_cost(logits, hparams):
  negativity_cost = tf.nn.relu(-logits)
  max_mask = tf.one_hot(tf.argmax(tf.abs(logits), -1), hparams.num_classes)
  min_logits = tf.abs(logits) * (1 - max_mask)
  max_logit = tf.abs(logits) * max_mask
  one_bound = tf.nn.relu(logits - hparams.logit_bound)
  axis_alignedness_cost = tf.nn.relu(min_logits - 0.1 * hparams.logit_bound)

  logits_packed = tf.reduce_all(tf.less(max_logit, hparams.logit_bound), -1)
  logits_packed = tf.logical_and(logits_packed,
                                 tf.reduce_all(
                                     tf.less(min_logits,
                                             0.1 * hparams.logit_bound), -1))
  logits_packed = tf.reduce_mean(tf.to_float(logits_packed))
  tf.summary.scalar("logits_packed", logits_packed)
  tf.summary.scalar(
      "logits_max",
      tf.to_float(tf.shape(max_logit)[-1]) * tf.reduce_mean(max_logit))

  return negativity_cost, axis_alignedness_cost, one_bound


def ard_cost():
  with tf.variable_scope("ard_cost"):
    cost = 0
    for v in tf.trainable_variables():
      if "kernel" in v.name or "DW" in v.name:
        rv = tf.reshape(v, [-1, int(v.shape[-1])])
        sq_rv = tf.square(rv)
        sum_sq = tf.reduce_sum(sq_rv, axis=1, keepdims=True)
        ard = sq_rv / (sum_sq / tf.cast(tf.shape(sq_rv)[1], tf.float32)
                      ) - 0.5 * tf.log(sum_sq)
        cost += tf.reduce_sum(ard)

    return cost


def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i, dim in enumerate(static):
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret


def standardize_images(x):
  """Image standardization on batches."""

  with tf.name_scope("standardize_images", [x]):
    x = tf.to_float(x)
    x_mean = tf.reduce_mean(x, axis=[1, 2, 3], keep_dims=True)
    x_variance = tf.reduce_mean(
        tf.square(x - x_mean), axis=[1, 2, 3], keep_dims=True)
    x_shape = shape_list(x)
    num_pixels = tf.to_float(x_shape[1] * x_shape[2] * x_shape[3])
    x = (x - x_mean) / tf.maximum(tf.sqrt(x_variance), tf.rsqrt(num_pixels))
    return x


def batch_norm(inputs, hparams, training):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  if hparams.data_format == "channels_first":
    axis = 1
  else:
    axis = -1

  return tf.layers.batch_normalization(
      inputs=inputs,
      axis=axis,
      momentum=0.997,
      epsilon=0.001,
      center=True,
      scale=True,
      training=training,
      fused=True)


def louizos_complexity_cost(params):
  gates = {
      w.name.strip(":0"): w
      for w in tf.trainable_variables()
      if "gates" in w.name
  }
  names = list(gates.keys())
  concat_gates = tf.concat([tf.reshape(gates[name], [-1]) for name in names],
                           0)
  if params.dropout_type == "louizos_weight":
    complexity_cost = tf.nn.sigmoid(
        concat_gates - params.louizos_beta * tf.
        log(-1 * params.louizos_gamma / params.louizos_zeta))
  elif params.dropout_type == "louizos_unit":
    reshaped_gates = [
        tf.reshape(gates[name], [-1, gates[name].shape[-1]]) for name in names
    ]

    parameters = []
    for name in names:
      g_name = name[:-len("louizos/gates")] + "DW"
      g = tf.contrib.framework.get_unique_variable(g_name)
      parameters.extend(
          [reduce(operator.mul,
                  g.shape.as_list()[:-1], 1)] * g.shape.as_list()[-1])
    group_sizes = tf.constant(parameters)
    assert group_sizes.shape[0] == concat_gates.shape[0], "{} != {}".format(
        group_sizes.shape[0], concat_gates.shape[0])

    complexity_cost = tf.cast(group_sizes, tf.float32) * tf.nn.sigmoid(
        concat_gates - params.louizos_beta * tf.
        log(-1 * params.louizos_gamma / params.louizos_zeta))
  return tf.reduce_sum(complexity_cost)


def switch_loss():
  losses = 0

  for v in tf.trainable_variables():
    if "switch" in v.name:
      losses += tf.reduce_sum(tf.abs(v))

  tf.summary.scalar("switch_loss", losses)
  return losses


def nonzero_count():
  nonzeroes = 0
  for op in tf.get_default_graph().get_operations():
    if "post_dropout" in op.name:
      v = tf.get_default_graph().get_tensor_by_name(op.name + ":0")
      count = tf.to_float(tf.equal(v, 0.))
      count = tf.reduce_sum(1 - count)
      nonzeroes += count
  return nonzeroes


def percent_sparsity():
  nonzeroes = 0
  total = 0
  for op in tf.get_default_graph().get_operations():
    if "post_dropout" in op.name:
      v = tf.get_default_graph().get_tensor_by_name(op.name + ":0")
      count = tf.to_float(tf.equal(v, 0.))
      count = tf.reduce_sum(1 - count)
      nonzeroes += count
      total += tf.size(v)
  return tf.to_float(nonzeroes) / tf.to_float(total)


def convert(num, base, length=None):
  ''' Converter from decimal to numeral systems from base 2 to base 10 '''
  num = int(num)
  base = int(base)
  result = []
  if num == 0:
    result.append(0)
  else:
    while (num > 0):
      result.append(num % base)
      num //= base
  # Reverse from LSB to MSB
  result = result[::-1]
  if length is not None:
    n_to_fill = length - len(result)
    if n_to_fill > 0:
      result = [0] * n_to_fill + result
  return result


def equal_mult(size, num_branches):
  return [
      tf.constant(1.0 / num_branches, shape=[size, 1, 1, 1], dtype=tf.float32)
      for _ in range(num_branches)
  ]


def uniform(size, num_branches):
  return [
      tf.random_uniform([size, 1, 1, 1], minval=0, maxval=1, dtype=tf.float32)
      for _ in range(num_branches)
  ]


def bernoulli(size, num_branches):
  random = tf.random_uniform([size], maxval=num_branches, dtype=tf.int32)
  bernoulli = tf.one_hot(random, depth=num_branches)
  rand = tf.split(bernoulli, [1] * num_branches, 1)
  rand = [tf.reshape(x, [-1, 1, 1, 1]) for x in rand]
  return rand


def combine(rand_uniform, rand_bernoulli, num_branches):
  return [
      tf.concat([rand_uniform[i], rand_bernoulli[i]], axis=0)
      for i in range(num_branches)
  ]


def model_top(labels, preds, cost, lr, mode, hparams):
  tf.summary.scalar("acc",
      tf.reduce_mean(
          tf.to_float(
              tf.equal(labels,
                       tf.argmax(
                           preds, axis=-1,
                           output_type=tf.int32)))))
  tf.summary.scalar("loss", cost)

  gs = tf.train.get_global_step()

  if hparams.weight_decay_and_noise:
    cost = weight_decay_and_noise(cost, hparams, lr)
    cost = tf.identity(cost, name="total_loss")
  optimizer = get_optimizer(lr, hparams)

  train_op = tf.contrib.layers.optimize_loss(
      name="training",
      loss=cost,
      global_step=gs,
      learning_rate=lr,
      clip_gradients=hparams.clip_grad_norm or None,
      gradient_noise_scale=hparams.grad_noise_scale or None,
      optimizer=optimizer,
      colocate_gradients_with_ops=True)

  if hparams.use_tpu:

    def metric_fn(l, p):
      return {
          "acc":
          tf.metrics.accuracy(
              labels=l, predictions=tf.argmax(p, -1, output_type=tf.int32)),
      }

    host_call = None
    if hparams.tpu_summarize:
      host_call = tpu.create_host_call(hparams.output_dir)
    tpu.remove_summaries()

    if mode == tf.estimator.ModeKeys.EVAL:
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          predictions=preds,
          loss=cost,
          eval_metrics=(metric_fn, [labels, preds]),
          host_call=host_call)

    return tpu_estimator.TPUEstimatorSpec(
        mode=mode, loss=cost, train_op=train_op, host_call=host_call)

  return tf.estimator.EstimatorSpec(
      mode,
      eval_metric_ops={
          "acc":
          tf.metrics.accuracy(
              labels=labels,
              predictions=tf.argmax(preds, axis=-1, output_type=tf.int32)),
      },
      loss=cost,
      train_op=train_op)
