import tensorflow as tf

from ..registry import register

from ..utils.activations import get_activation
from ..utils.dropouts import get_dropout
from ..utils.initializations import get_init
from ..utils.optimizers import get_optimizer
from ..utils import model_utils


@register("lenet")
def get_lenet(hparams, lr):
  """Callable model function compatible with Experiment API.

    Args:
      params: a HParams object containing values for fields:
      lr: learning rate variable
    """

  def _conv(name, x, filter_size, in_filters, out_filters, strides, mode):
    """Convolution."""
    with tf.variable_scope(name):
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32)
      is_training = mode == tf.estimator.ModeKeys.TRAIN
      if hparams.dropout_type is not None:
        dropout_fn = get_dropout(hparams.dropout_type)
        kernel = dropout_fn(kernel, hparams, is_training)

        # special case for variational
        if hparams.dropout_type and "variational" in hparams.dropout_type:
          kernel, log_alpha = kernel[0], kernel[1]
          if is_training:
            conved_mu = tf.nn.conv2d(
                x, kernel, strides=strides, padding='VALID')
            conved_si = tf.sqrt(
                tf.nn.conv2d(
                    tf.square(x),
                    tf.exp(log_alpha) * tf.square(kernel),
                    strides=strides,
                    padding='VALID') + 1e-8)
            return conved_mu + tf.random_normal(
                tf.shape(conved_mu)) * conved_si, tf.count_nonzero(kernel)

      return tf.nn.conv2d(x, kernel, strides, padding='VALID')

  def lenet(features, labels, mode, params):
    """The lenet neural net net template.

            Args:
              features: a dict containing key "inputs"
              mode: training, evaluation or infer
            """
    with tf.variable_scope("lenet", initializer=get_init(hparams)):
      is_training = mode == tf.estimator.ModeKeys.TRAIN
      actvn = get_activation(hparams)

      if hparams.use_tpu and 'batch_size' in params.keys():
        hparams.batch_size = params['batch_size']

      # input layer
      x = features["inputs"]
      x = model_utils.standardize_images(x)

      # unflatten
      x = tf.reshape(x, [hparams.batch_size] + hparams.input_shape)

      # conv1
      b_conv1 = tf.get_variable(
          "Variable", initializer=tf.constant_initializer(0.1), shape=[6])
      h_conv1 = _conv('conv1', x, 5, 3, 6, [1, 1, 1, 1], mode) + b_conv1
      h_conv1 = tf.nn.relu(h_conv1)
      h_pool1 = tf.nn.max_pool(
          h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

      # conv2
      b_conv2 = tf.get_variable(
          "Variable_1", initializer=tf.constant_initializer(0.1), shape=[16])
      h_conv2 = _conv('conv2', h_pool1, 5, 6, 16, [1, 1, 1, 1], mode) + b_conv2
      h_conv2 = tf.nn.relu(h_conv2)
      h_pool2 = tf.nn.max_pool(
          h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

      # flatten for fc
      h_pool2_flat = tf.reshape(h_pool2, [hparams.batch_size, -1])

      # fc1
      with tf.variable_scope('fc1'):
        h_fc1 = tf.nn.relu(
            model_utils.dense(h_pool2_flat, 500, hparams, is_training))

      # fc2
      with tf.variable_scope('fc2'):
        y = model_utils.dense(h_fc1, 10, hparams, is_training, dropout=False)

      if mode in [model_utils.ModeKeys.PREDICT, model_utils.ModeKeys.ATTACK]:
        predictions = {
            'classes': tf.argmax(y, axis=1),
            'logits': y,
            'probabilities': tf.nn.softmax(y, name='softmax_tensor'),
        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=y)

      if hparams.axis_aligned_cost:
        negativity_cost, axis_alignedness_cost, one_bound = model_utils.axis_aligned_cost(
            y, hparams)
        masked_max = tf.abs(y) * (
            1 - tf.one_hot(tf.argmax(tf.abs(y), -1), hparams.num_classes))
        tf.summary.scalar(
            "logit_prior",
            tf.reduce_mean(
                tf.to_float(
                    tf.logical_and(masked_max >= 0.0, masked_max <= 0.1))))
        tf.summary.scalar("avg_max",
                          tf.reduce_mean(tf.reduce_max(tf.abs(y), axis=-1)))
        loss += hparams.axis_aligned_cost * tf.reduce_mean(
            negativity_cost + axis_alignedness_cost + 20. * one_bound)

      if hparams.logit_squeezing:
        loss += hparams.logit_squeezing * tf.reduce_mean(y**2)

      if hparams.clp:
        loss += hparams.clp * tf.reduce_mean(
            (y[:hparams.batch_size // 2] - y[hparams.batch_size // 2:])**2)

      if hparams.dropout_type and "variational" in hparams.dropout_type:
        # prior DKL part of the ELBO
        graph = tf.get_default_graph()
        node_defs = [
            n for n in graph.as_graph_def().node if 'log_alpha' in n.name
        ]
        log_alphas = [
            graph.get_tensor_by_name(n.name + ":0") for n in node_defs
        ]
        divergences = [model_utils.dkl_qp(la) for la in log_alphas]
        # combine to form the ELBO
        N = float(50000)
        dkl = tf.reduce_sum(tf.stack(divergences))

        warmup_steps = 50000
        inv_base = tf.exp(tf.log(0.01) / warmup_steps)
        inv_decay = inv_base**(
            warmup_steps - tf.to_float(tf.train.get_global_step()))

        loss += (1. / N) * dkl * inv_decay * hparams.var_scale

      if hparams.smallify > 0.0:
        loss += model_utils.switch_loss() * hparams.smallify

      return model_utils.model_top(labels, tf.nn.softmax(y, -1), loss, lr,
                                   mode, hparams)

  return lenet
