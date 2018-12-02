import tensorflow as tf
import numpy as np

from ..utils import dropouts
from ..utils.activations import get_activation
from ..utils.dropouts import get_dropout, smallify_dropout
from ..utils.initializations import get_init
from ..registry import register
from ..utils import model_utils
from ..utils.model_utils import ModeKeys
from ...training import tpu


@register("resnet")
def get_resnet(hparams, lr):
  """Callable model function compatible with Experiment API.

          Args:
            params: a HParams object containing values for fields:
              use_bottleneck: bool to bottleneck the network
              num_residual_units: number of residual units
              num_classes: number of classes
              batch_size: batch size
              weight_decay_rate: weight decay rate
          """

  def resnet(features, labels, mode, params):
    if hparams.use_tpu and 'batch_size' in params.keys():
      hparams.batch_size = params['batch_size']

    is_training = mode == tf.estimator.ModeKeys.TRAIN

    def _residual(x, out_filter, stride, projection=False):
      """Residual unit with 2 sub layers."""
      is_variational = hparams.dropout_type is not None and "variational" in hparams.dropout_type

      orig_x = x
      if not is_variational:
        x = model_utils.batch_norm(x, hparams, is_training)
        x = tf.nn.relu(x)

      if projection:
        orig_x = model_utils.conv(
            x,
            1,
            out_filter,
            hparams,
            is_training=is_training,
            strides=stride,
            name="shortcut")

      with tf.variable_scope('sub1'):
        x = model_utils.conv(
            x,
            3,
            out_filter,
            hparams,
            is_training=is_training,
            strides=stride,
            name='conv1')

        x = model_utils.batch_norm(x, hparams, is_training)
        x = tf.nn.relu(x)

      with tf.variable_scope('sub2'):
        x = model_utils.conv(
            x,
            3,
            out_filter,
            hparams,
            is_training=is_training,
            strides=[1, 1, 1, 1],
            name='conv2')

      x += orig_x

      return x

    def _bottleneck_residual(x, out_filter, stride, projection=False):
      """Residual unit with 3 sub layers."""

      is_variational = hparams.dropout_type is not None and "variational" in hparams.dropout_type

      orig_x = x
      if not is_variational:
        x = model_utils.batch_norm(x, hparams, is_training)
        x = tf.nn.relu(x)

      if projection:
        orig_x = model_utils.conv(
            x,
            1,
            4 * out_filter,
            hparams,
            is_training=is_training,
            strides=stride,
            name="shortcut")

      with tf.variable_scope('sub1'):
        x = model_utils.conv(
            x,
            1,
            out_filter,
            hparams,
            is_training=is_training,
            strides=[1, 1, 1, 1],
            name='conv1')
        x = model_utils.batch_norm(x, hparams, is_training)
        x = tf.nn.relu(x)
      with tf.variable_scope('sub2'):
        x = model_utils.conv(
            x,
            3,
            out_filter,
            hparams,
            is_training=is_training,
            strides=stride,
            name='conv2')
        x = model_utils.batch_norm(x, hparams, is_training)
        x = tf.nn.relu(x)
      with tf.variable_scope('sub3'):
        x = model_utils.conv(
            x,
            1,
            4 * out_filter,
            hparams,
            is_training=is_training,
            strides=[1, 1, 1, 1],
            name='conv3')

      return orig_x + x

    def _l1():
      """L1 weight decay loss."""
      if hparams.l1_norm == 0:
        return 0

      costs = []
      for var in tf.trainable_variables():
        if "DW" in var.name and "logit" not in var.name:
          costs.append(tf.reduce_mean(tf.abs(var)))

      return tf.multiply(hparams.l1_norm, tf.add_n(costs))

    def _fully_connected(x, out_dim):
      """FullyConnected layer for final output."""
      prev_dim = np.product(x.get_shape().as_list()[1:])
      x = tf.reshape(x, [hparams.batch_size, prev_dim])
      w = tf.get_variable('DW', [prev_dim, out_dim])
      b = tf.get_variable(
          'biases', [out_dim], initializer=tf.zeros_initializer())
      return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(x):
      assert x.get_shape().ndims == 4
      if hparams.data_format == "channels_last":
        return tf.reduce_mean(x, [1, 2])

      return tf.reduce_mean(x, [2, 3])

    def _stride_arr(stride):
      """Map a stride scalar to the stride array for tf.nn.conv2d."""
      if hparams.data_format == "channels_last":
        return [1, stride, stride, 1]

      return [1, 1, stride, stride]

    if mode == ModeKeys.PREDICT or mode == ModeKeys.ATTACK:
      if "labels" in features:
        labels = features["labels"]

    with tf.variable_scope("resnet", initializer=get_init(hparams)):
      hparams.mode = mode
      strides = [1, 2, 2, 2]
      res_func = (_residual
                  if not hparams.use_bottleneck else _bottleneck_residual)
      filters = hparams.residual_filters
      large_input = hparams.input_shape[0] > 32

      # 3 and 16 picked from example implementation
      with tf.variable_scope('init'):
        x = features["inputs"]
        stride = _stride_arr(2) if large_input else _stride_arr(1)
        x = model_utils.conv(
            x,
            7,
            filters[0],
            hparams,
            strides=stride,
            dropout=False,
            name='init_conv')

        if large_input:
          x = tf.layers.max_pooling2d(
              inputs=x,
              pool_size=3,
              strides=2,
              padding="SAME",
              data_format=hparams.data_format)

      with tf.variable_scope('unit_1_0'):
        x = res_func(x, filters[1], _stride_arr(strides[0]), True)

      for i in range(1, hparams.residual_units[0]):
        with tf.variable_scope('unit_1_%d' % i):
          x = res_func(x, filters[1], _stride_arr(1), False)

      with tf.variable_scope('unit_2_0'):
        x = res_func(x, filters[2], _stride_arr(strides[1]), True)

      for i in range(1, hparams.residual_units[1]):
        with tf.variable_scope('unit_2_%d' % i):
          x = res_func(x, filters[2], _stride_arr(1), False)

      with tf.variable_scope('unit_3_0'):
        x = res_func(x, filters[3], _stride_arr(strides[2]), True)

      for i in range(1, hparams.residual_units[2]):
        with tf.variable_scope('unit_3_%d' % i):
          x = res_func(x, filters[3], _stride_arr(1), False)

      if len(filters) == 5:
        with tf.variable_scope('unit_4_0'):
          x = res_func(x, filters[4], _stride_arr(strides[3]), True)

        for i in range(1, hparams.residual_units[3]):
          with tf.variable_scope('unit_4_%d' % i):
            x = res_func(x, filters[4], _stride_arr(1), False)

      x = model_utils.batch_norm(x, hparams, is_training)
      x = tf.nn.relu(x)

      with tf.variable_scope('unit_last'):
        x = _global_avg_pool(x)

      with tf.variable_scope('logit'):
        logits = _fully_connected(x, hparams.num_classes)
        predictions = tf.nn.softmax(logits)

      if mode in [ModeKeys.PREDICT, ModeKeys.ATTACK]:

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'classes': tf.argmax(predictions, axis=1),
                'logits': logits,
                'probabilities': predictions,
            })

      with tf.variable_scope('costs'):
        xent = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)
        cost = tf.reduce_mean(xent, name='xent')
        if is_training:
          cost += model_utils.weight_decay(hparams)
          cost += _l1()

          if hparams.dropout_type is not None:
            if "louizos" in hparams.dropout_type:
              cost += hparams.louizos_cost * model_utils.louizos_complexity_cost(
                  hparams) / 50000

            if "variational" in hparams.dropout_type:
              # prior DKL part of the ELBO
              graph = tf.get_default_graph()
              node_defs = [
                  n for n in graph.as_graph_def().node if 'log_alpha' in n.name
              ]
              log_alphas = [
                  graph.get_tensor_by_name(n.name + ":0") for n in node_defs
              ]
              print([
                  n.name
                  for n in graph.as_graph_def().node
                  if 'log_alpha' in n.name
              ])
              print("found %i logalphas" % len(log_alphas))
              divergences = [dropouts.dkl_qp(la) for la in log_alphas]
              # combine to form the ELBO
              N = float(50000)
              dkl = tf.reduce_sum(tf.stack(divergences))

              warmup_steps = 50000
              dkl = (1. / N) * dkl * tf.minimum(
                  1.0,
                  tf.to_float(tf.train.get_global_step()) /
                  warmup_steps) * hparams.var_scale
              cost += dkl
              tf.summary.scalar("dkl", dkl)

          if hparams.ard_cost > 0.0:
            cost += model_utils.ard_cost() * hparams.ard_cost

          if hparams.smallify > 0.0:
            cost += model_utils.switch_loss() * hparams.smallify

    # Summaries
    # ========================
    tf.summary.scalar("total_nonzero", model_utils.nonzero_count())
    all_weights = tf.concat(
        [
            tf.reshape(v, [-1])
            for v in tf.trainable_variables()
            if "DW" in v.name
        ],
        axis=0)
    tf.summary.histogram("weights", all_weights)
    # ========================

    return model_utils.model_top(labels, predictions, cost, lr, mode, hparams)

  return resnet
