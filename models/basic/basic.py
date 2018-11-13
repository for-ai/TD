import tensorflow as tf

from ..registry import register

from ..utils.activations import get_activation
from ..utils.initializations import get_init
from ..utils.optimizers import get_optimizer
from ..utils import model_utils


@register("basic")
def get_basic(params, lr):
  """Callable model function compatible with Experiment API.

  Args:
    params: a HParams object containing values for fields:
    lr: learning rate variable
  """

  def basic(features, labels, mode, _):
    """The basic neural net net template.

    Args:
      features: a dict containing key "inputs"
      mode: training, evaluation or infer
    """
    with tf.variable_scope("basic", initializer=get_init(params)):
      is_training = mode == tf.estimator.ModeKeys.TRAIN
      actvn = get_activation(params)
      x = features["inputs"]
      batch_size = tf.shape(x)[0]

      nonzero = 0
      activations = []
      for i, feature_count in enumerate(params.layers):
        with tf.variable_scope("layer_%d" % i):
          if params.layer_type == "dense":
            x, w = model_utils.collect_vars(
                lambda: model_utils.dense(x, feature_count, params, is_training)
            )
          elif params.layer_type == "conv":
            x, w = model_utils.collect_vars(lambda: tf.layers.conv2d(
                x, feature_count, params.kernel_size, padding="SAME"))
          if params.batch_norm:
            x = tf.layers.batch_normalization(x, training=is_training)
          x = actvn(x)
          activations.append(x)
      x = tf.reshape(x, [batch_size, params.layers[-1]])
      with tf.variable_scope('logit'):
        x = tf.layers.dense(x, params.output_shape[0], use_bias=False)

      if mode in [model_utils.ModeKeys.PREDICT, model_utils.ModeKeys.ATTACK]:
        predictions = {
            'classes': tf.argmax(x, axis=1),
            'logits': x,
            'probabilities': tf.nn.softmax(x, name='softmax_tensor'),
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=x)
      if params.smallify > 0.0:
        loss += model_utils.switch_loss() * params.smallify

      # Summaries
      # ========================
      if not params.use_tpu:
        tf.summary.scalar("nonzero", model_utils.nonzero_count())
        tf.summary.scalar("percent_sparsity", model_utils.percent_sparsity())
      # ========================

      return model_utils.model_top(labels, tf.nn.softmax(x, -1), loss, lr,
                                   mode, params)

  return basic
