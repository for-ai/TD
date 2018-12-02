import tensorflow as tf
import os
import numpy as np

from ...hparams.registry import get_hparams
from ...models.registry import get_model
from ...data.registry import get_input_fns
from ...training import flags
from .prune import get_prune_fn, get_current_weights, get_louizos_masks, get_smallify_masks, prune_weights, is_prunable_weight


def init_flags():
  tf.flags.DEFINE_string("model", None, "Which model to use.")
  tf.flags.DEFINE_string("data", None, "Which data to use.")
  tf.flags.DEFINE_string("env", None, "Which environment to use.")
  tf.flags.DEFINE_string("hparams", None, "Which hparams to use.")
  tf.flags.DEFINE_string("hparam_override", "",
                         "Run-specific hparam settings to use.")
  tf.flags.DEFINE_string("output_dir", None, "The output directory.")
  tf.flags.DEFINE_string("data_dir", None, "The data directory.")
  tf.flags.DEFINE_integer("train_steps", 10000,
                          "Number of training steps to perform.")
  tf.flags.DEFINE_integer("eval_steps", 100,
                          "Number of evaluation steps to perform.")
  tf.flags.DEFINE_integer("eval_every", 1000,
                          "Number of steps between evaluations.")
  tf.flags.DEFINE_string(
      "post_weights_dir", "",
      "folder of the weights, if not set defaults to output_dir")
  tf.flags.DEFINE_string("prune_percent", "0.5",
                         "percent of weights to prune, comma separated")
  tf.flags.DEFINE_string("prune", "one_shot", "one_shot or fisher")
  tf.flags.DEFINE_boolean("variational", False, "use evaluate")
  tf.flags.DEFINE_string("eval_file", "eval_prune_results",
                         "file to put results")
  tf.flags.DEFINE_integer("train_epochs", None,
                          "Number of training epochs to perform.")
  tf.flags.DEFINE_integer("eval_steps", None,
                          "Number of evaluation steps to perform.")

def eval_model(FLAGS, hparam_name):
  hparams = get_hparams(hparam_name)
  hparams = hparams.parse(FLAGS.hparam_override)
  hparams = flags.update_hparams(FLAGS, hparams)

  model_fn = get_model(hparams)
  _, _, test_input_fn = get_input_fns(hparams, generate=False)

  features, labels = test_input_fn()
  sess = tf.Session()
  tf.train.create_global_step()
  model_fn(features, labels, tf.estimator.ModeKeys.TRAIN)
  saver = tf.train.Saver()
  ckpt_dir = tf.train.latest_checkpoint(hparams.output_dir)
  print("Loading model from...", ckpt_dir)
  saver.restore(sess, ckpt_dir)

  evals = []
  prune_percents = [float(i) for i in FLAGS.prune_percent.split(",")]

  mode = "standard"
  orig_weights = get_current_weights(sess)
  louizos_masks, smallify_masks = None, None
  if "louizos" in hparam_name:
    louizos_masks = get_louizos_masks(sess, orig_weights)
    mode = "louizos"
  elif "smallify" in hparam_name:
    smallify_masks = get_smallify_masks(sess, orig_weights)
  elif "variational" in hparam_name:
    mode = "variational"

  for prune_percent in prune_percents:
    if prune_percent > 0.0:
      prune_fn = get_prune_fn(FLAGS.prune)(mode, k=prune_percent)
      w_copy = dict(orig_weights)
      sm_copy = dict(smallify_masks) if smallify_masks is not None else None
      lm_copy = dict(louizos_masks) if louizos_masks is not None else None
      post_weights_pruned, weight_counts = prune_weights(
          prune_fn,
          w_copy,
          louizos_masks=lm_copy,
          smallify_masks=sm_copy,
          hparams=hparams)
      print("current weight counts at {}: {}".format(prune_percent,
                                                     weight_counts))

      print("there are ", len(tf.trainable_variables()), " weights")
      for v in tf.trainable_variables():
        if is_prunable_weight(v):
          assign_op = v.assign(
              np.reshape(post_weights_pruned[v.name.strip(":0")], v.shape))
          sess.run(assign_op)

    saver.save(sess, os.path.join(hparams.output_dir, "tmp", "model"))
    estimator = tf.estimator.Estimator(
        model_fn=tf.contrib.estimator.replicate_model_fn(model_fn),
        model_dir=os.path.join(hparams.output_dir, "tmp"))
    print(
        f"Processing pruning {prune_percent} of weights for {hparams.eval_steps} steps"
    )  
    acc = estimator.evaluate(test_input_fn, hparams.eval_steps)['acc']
    print(f"Accuracy @ prune {100*prune_percent}% is {acc}")
    evals.append(acc)
  return evals


def _run(FLAGS):
  eval_file = open(FLAGS.eval_file, "w")

  hparams_list = FLAGS.hparams.split(",")
  total_evals = {}
  for hparam_name in hparams_list:
    evals = eval_model(FLAGS, hparam_name)

    print(hparam_name, ":", evals)
    eval_file.writelines("{}:{}\n".format(hparam_name, evals))
    total_evals[hparam_name] = evals
    tf.reset_default_graph()

  print("processed results:", total_evals)
  eval_file.close()


if __name__ == "__main__":
  init_flags()
  FLAGS = tf.app.flags.FLAGS
  _run(FLAGS)
