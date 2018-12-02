import cloud
import os
import sys
import subprocess
import random
import tensorflow as tf
import numpy as np
import time
import logging

from .hparams.registry import get_hparams
from .models.registry import get_model
from .data.registry import get_input_fns
from .training.lr_schemes import get_lr
from .training.envs import get_env
from .training import flags
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator


def init_flags():
  tf.flags.DEFINE_string("env", None, "Which environment to use.")  # required
  tf.flags.DEFINE_string("hparams", None, "Which hparams to use.")  # required
  # Utility flags
  tf.flags.DEFINE_string("hparam_override", "",
                         "Run-specific hparam settings to use.")
  tf.flags.DEFINE_boolean("fresh", False, "Remove output_dir before running.")
  tf.flags.DEFINE_integer("seed", None, "Random seed.")
  tf.flags.DEFINE_integer("train_epochs", None,
                          "Number of training epochs to perform.")
  tf.flags.DEFINE_integer("eval_steps", None,
                          "Number of evaluation steps to perform.")
  # TPU flags
  tf.flags.DEFINE_string("tpu_name", "", "Name of TPU(s)")
  tf.flags.DEFINE_integer(
      "tpu_iterations_per_loop", 1000,
      "The number of training steps to run on TPU before"
      "returning control to CPU.")
  tf.flags.DEFINE_integer(
      "tpu_shards", 8, "The number of TPU shards in the system "
      "(a single Cloud TPU has 8 shards.")
  tf.flags.DEFINE_boolean(
      "tpu_summarize", False, "Save summaries for TensorBoard. "
      "Warning: this will slow down execution.")
  tf.flags.DEFINE_boolean("tpu_dedicated", False,
                          "Do not use preemptible TPUs.")
  tf.flags.DEFINE_string("data_dir", None, "The data directory.")
  tf.flags.DEFINE_string("output_dir", None, "The output directory.")
  tf.flags.DEFINE_integer("eval_every", 1000,
                          "Number of steps between evaluations.")


tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = None


def init_random_seeds():
  tf.set_random_seed(FLAGS.seed)
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)


def init_model(hparams_name):
  flags.validate_flags(FLAGS)

  tf.reset_default_graph()

  hparams = get_hparams(hparams_name)
  hparams = hparams.parse(FLAGS.hparam_override)
  hparams = flags.update_hparams(FLAGS, hparams, hparams_name)

  # set larger eval_every for TPUs to improve utilization
  if FLAGS.env == "tpu":
    FLAGS.eval_every = max(FLAGS.eval_every, 5000)
    hparams.tpu_summarize = FLAGS.tpu_summarize

  tf.logging.warn("\n-----------------------------------------\n"
                  "BEGINNING RUN:\n"
                  "\t hparams: %s\n"
                  "\t output_dir: %s\n"
                  "\t data_dir: %s\n"
                  "-----------------------------------------\n" %
                  (hparams_name, hparams.output_dir, hparams.data_dir))

  return hparams


def construct_estimator(model_fn, hparams, tpu=None):
  if hparams.use_tpu:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu=tpu.name)
    master = tpu_cluster_resolver.get_master()
    config = tpu_config.RunConfig(
        master=master,
        evaluation_master=master,
        model_dir=hparams.output_dir,
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True),
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=FLAGS.tpu_iterations_per_loop,
            num_shards=FLAGS.tpu_shards),
        save_checkpoints_steps=FLAGS.eval_every)
    estimator = tpu_estimator.TPUEstimator(
        use_tpu=hparams.use_tpu,
        model_fn=model_fn,
        model_dir=hparams.output_dir,
        config=config,
        train_batch_size=hparams.batch_size,
        eval_batch_size=hparams.batch_size)
  else:
    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=FLAGS.eval_every, session_config=gpu_config)

    estimator = tf.estimator.Estimator(
        model_fn=tf.contrib.estimator.replicate_model_fn(model_fn),
        model_dir=hparams.output_dir,
        config=run_config)

  return estimator


def _run(hparams_name):
  """Run training, evaluation and inference."""
  hparams = init_model(hparams_name)
  original_batch_size = hparams.batch_size
  if tf.gfile.Exists(hparams.output_dir) and FLAGS.fresh:
    tf.gfile.DeleteRecursively(hparams.output_dir)

  if not tf.gfile.Exists(hparams.output_dir):
    tf.gfile.MakeDirs(hparams.output_dir)
  model_fn = get_model(hparams)
  train_input_fn, eval_input_fn, test_input_fn = get_input_fns(hparams)

  tpu = None
  if hparams.use_tpu:
    cloud.instance.tpu.clean()
    tpu = cloud.instance.tpu.get(preemptible=not FLAGS.tpu_dedicated)

  estimator = construct_estimator(model_fn, hparams, tpu)

  if not hparams.use_tpu:
    features, labels = train_input_fn()
    sess = tf.Session()
    tf.train.get_or_create_global_step()

    model_fn(features, labels, tf.estimator.ModeKeys.TRAIN)
    sess.run(tf.global_variables_initializer())

  # output metadata about the run
  with tf.gfile.GFile(os.path.join(hparams.output_dir, 'hparams.txt'),
                      'w') as hparams_file:
    hparams_file.write("{}\n".format(time.time()))
    hparams_file.write("{}\n".format(str(hparams)))

  def loop(steps=FLAGS.eval_every):
    estimator.train(train_input_fn, steps=steps)
    if eval_input_fn:
      estimator.evaluate(eval_input_fn, steps=hparams.eval_steps, name="eval")
    if test_input_fn:
      estimator.evaluate(test_input_fn, steps=hparams.eval_steps, name="test")

  loop(1)

  steps = estimator.get_variable_value("global_step")
  k = steps * original_batch_size / float(hparams.epoch_size)
  while k <= hparams.train_epochs:
    tf.logging.info("Beginning epoch %f / %d" % (k, hparams.train_epochs))

    if tpu and not tpu.usable:
      tpu.delete(async=True)
      tpu = cloud.instance.tpu.get(preemptible=not FLAGS.tpu_dedicated)
      estimator = construct_estimator(model_fn, hparams, tpu)

    loop()

    steps = estimator.get_variable_value("global_step")
    k = steps * original_batch_size / float(hparams.epoch_size)


def main(_):
  global FLAGS
  FLAGS = tf.app.flags.FLAGS

  init_random_seeds()
  if FLAGS.env != "local":
    cloud.connect()
  for hparams_name in FLAGS.hparams.split(","):
    _run(hparams_name)


if __name__ == "__main__":
  init_flags()
  tf.app.run()
