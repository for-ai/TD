import getpass
import os
import subprocess

import tensorflow as tf

from .envs import get_env


def validate_flags(FLAGS):
  messages = []
  if not FLAGS.env:
    messages.append("Missing required flag --env")
  if not FLAGS.hparams:
    messages.append("Missing required flag --hparams")

  if len(messages) > 0:
    raise Exception("\n".join(messages))

  return FLAGS


def update_hparams(FLAGS, hparams, hparams_name):
  hparams.env = FLAGS.env
  hparams.use_tpu = hparams.env == "tpu"
  hparams.train_epochs = FLAGS.train_epochs or hparams.train_epochs
  hparams.eval_steps = FLAGS.eval_steps or hparams.eval_steps

  env = get_env(FLAGS.env)
  hparams.data_dir = os.path.join(FLAGS.data_dir or env.data_dir, hparams.data)
  hparams.output_dir = os.path.join(env.output_dir, FLAGS.hparams)

  return hparams
