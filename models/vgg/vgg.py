import tensorflow as tf

from ..utils.activations import get_activation
from ..utils.dropouts import get_dropout
from ..utils.initializations import get_init
from ..utils.optimizers import get_optimizer
from ..registry import register
from ..utils import model_utils
from ..utils import dropouts
from ...training import tpu
import six

import numpy as np
from tensorflow.contrib.tpu.python.tpu import tpu_estimator, tpu_optimizer


def metric_fn(labels, predictions):
  return {
      "acc":
      tf.metrics.accuracy(
          labels=tf.argmax(labels, -1), predictions=tf.argmax(predictions,
                                                              -1)),
  }


@register("vgg")
def get_vgg(hparams, lr):
  """Callable model function compatible with Experiment API."""

  def vgg(features, labels, mode, params):
    if hparams.use_tpu and 'batch_size' in params.keys():
      hparams.batch_size = params['batch_size']

    is_training = mode == tf.estimator.ModeKeys.TRAIN

    inputs = features["inputs"]
    with tf.variable_scope("vgg", initializer=get_init(hparams)):
      total_nonzero = 0
      conv1_1 = model_utils.conv(
          inputs, 3, 64, hparams, name="conv1_1", is_training=is_training)

      conv1_1 = model_utils.batch_norm(conv1_1, hparams, is_training)
      conv1_1 = tf.nn.relu(conv1_1)

      conv1_2 = model_utils.conv(
          conv1_1, 3, 64, hparams, name="conv1_2", is_training=is_training)
      conv1_2 = model_utils.batch_norm(conv1_2, hparams, is_training)
      conv1_2 = tf.nn.relu(conv1_2)

      pool1 = tf.layers.max_pooling2d(
          conv1_2, 2, 2, padding="SAME", name='pool1')

      conv2_1 = model_utils.conv(
          pool1, 3, 128, hparams, name="conv2_1", is_training=is_training)
      conv2_1 = model_utils.batch_norm(conv2_1, hparams, is_training)
      conv2_1 = tf.nn.relu(conv2_1)

      conv2_2 = model_utils.conv(
          conv2_1, 3, 128, hparams, name="conv2_2", is_training=is_training)
      conv2_2 = model_utils.batch_norm(conv2_2, hparams, is_training)
      conv2_2 = tf.nn.relu(conv2_2)

      pool2 = tf.layers.max_pooling2d(
          conv2_2, 2, 2, padding="SAME", name='pool2')

      conv3_1 = model_utils.conv(
          pool2, 3, 256, hparams, name="conv3_1", is_training=is_training)
      conv3_1 = model_utils.batch_norm(conv3_1, hparams, is_training)
      conv3_1 = tf.nn.relu(conv3_1)

      conv3_2 = model_utils.conv(
          conv3_1, 3, 256, hparams, name="conv3_2", is_training=is_training)
      conv3_2 = model_utils.batch_norm(conv3_2, hparams, is_training)
      conv3_2 = tf.nn.relu(conv3_2)

      conv3_3 = model_utils.conv(
          conv3_2, 3, 256, hparams, name="conv3_3", is_training=is_training)
      conv3_3 = model_utils.batch_norm(conv3_3, hparams, is_training)
      conv3_3 = tf.nn.relu(conv3_3)

      pool3 = tf.layers.max_pooling2d(
          conv3_3, 2, 2, padding="SAME", name='pool3')

      conv4_1 = model_utils.conv(
          pool3, 3, 512, hparams, name="conv4_1", is_training=is_training)
      conv4_1 = model_utils.batch_norm(conv4_1, hparams, is_training)
      conv4_1 = tf.nn.relu(conv4_1)

      conv4_2 = model_utils.conv(
          conv4_1, 3, 512, hparams, name="conv4_2", is_training=is_training)
      conv4_2 = model_utils.batch_norm(conv4_2, hparams, is_training)
      conv4_2 = tf.nn.relu(conv4_2)

      conv4_3 = model_utils.conv(
          conv4_2, 3, 512, hparams, name="conv4_3", is_training=is_training)
      conv4_3 = model_utils.batch_norm(conv4_3, hparams, is_training)
      conv4_3 = tf.nn.relu(conv4_3)

      pool4 = tf.layers.max_pooling2d(
          conv4_3, 2, 2, padding="SAME", name='pool4')

      conv5_1 = model_utils.conv(
          pool4, 3, 512, hparams, name="conv5_1", is_training=is_training)
      conv5_1 = model_utils.batch_norm(conv5_1, hparams, is_training)
      conv5_1 = tf.nn.relu(conv5_1)

      conv5_2 = model_utils.conv(
          conv5_1, 3, 512, hparams, name="conv5_2", is_training=is_training)
      conv5_2 = model_utils.batch_norm(conv5_2, hparams, is_training)
      conv5_2 = tf.nn.relu(conv5_2)

      conv5_3 = model_utils.conv(
          conv5_2, 3, 512, hparams, name="conv5_3", is_training=is_training)
      conv5_3 = model_utils.batch_norm(conv5_3, hparams, is_training)
      conv5_3 = tf.nn.relu(conv5_3)

      pool5 = tf.layers.max_pooling2d(
          conv5_3, 2, 2, padding="SAME", name='pool5')

      flat_x = tf.reshape(pool5, [hparams.batch_size, 512])
      fc6 = model_utils.batch_norm(
          model_utils.dense(flat_x, 4096, hparams, is_training), hparams,
          is_training)
      fc7 = model_utils.batch_norm(
          model_utils.dense(fc6, 4096, hparams, is_training), hparams,
          is_training)

      logits = tf.layers.dense(fc7, hparams.num_classes, name="logits")
      probs = tf.nn.softmax(logits, axis=-1)

      if mode in [model_utils.ModeKeys.PREDICT, model_utils.ModeKeys.ATTACK]:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'classes': tf.argmax(probs, axis=1),
                'logits': logits,
                'probabilities': probs,
            })

      xent = tf.losses.sparse_softmax_cross_entropy(
          labels=labels, logits=logits)
      cost = tf.reduce_mean(xent, name='xent')
      cost += model_utils.weight_decay(hparams)

      tf.summary.scalar("total_nonzero", model_utils.nonzero_count())
      tf.summary.scalar("percent_sparsity", model_utils.percent_sparsity())
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

    return model_utils.model_top(labels, probs, cost, lr, mode, hparams)

  return vgg
