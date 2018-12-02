import numpy as np
import tensorflow as tf
import statistics
from ...models.utils import model_utils

_PRUNE_FN = dict()


def register(fn):
  global _PRUNE_FN
  _PRUNE_FN[fn.__name__] = fn
  return fn


def get_prune_fn(name):
  return _PRUNE_FN[name]


@register
def weight(mode, k=0.5):

  if mode == "standard":

    def prune(weight_dict, weight_key):
      weights = weight_dict[weight_key]
      w = weights.copy()
      if len(weights.shape) == 4:
        w = w.reshape([-1, weights.shape[-1]])

      abs_w = np.abs(w)
      idx = int(k * abs_w.shape[0])
      med = np.sort(abs_w, axis=0)[idx:idx + 1]
      mask = (abs_w >= med).astype(float)
      pruned_w = mask * w

      return pruned_w, mask
  elif mode == "variational":

    def prune(weight_dict, weight_key):
      weights = weight_dict[weight_key]
      if k == 0.0:
        return weights, None
      log_alpha = weight_dict[weight_key.strip("DW") + "variational/log_alpha"]
      w = weights.copy()
      la = log_alpha.copy()
      if len(weights.shape) == 4:
        w = w.reshape([-1, weights.shape[-1]])
        la = la.reshape([-1, weights.shape[-1]])

      idx = int((1 - k) * la.shape[0])
      med = np.sort(la, axis=0)[idx:idx + 1]
      mask = (la < med).astype(float)
      pruned_w = mask * w

      return pruned_w, mask
  elif mode == "louizos":

    def prune(weight_dict, weight_key):
      weights = weight_dict[weight_key]
      w = weights.copy()
      if len(weights.shape) == 4:
        w = w.reshape([-1, weights.shape[-1]])

      idx = int(k * w.shape[0])
      med = np.sort(w, axis=0)[idx:idx + 1]
      mask = (w >= med).astype(float)
      pruned_w = mask * w

      return pruned_w, mask

  return prune


@register
def unit(mode, k=0.5):

  if mode == "standard" or mode == "variational":

    def prune(weight_dict, weight_key):
      weights = weight_dict[weight_key]
      w = weights.copy()
      if len(weights.shape) == 4:
        w = w.reshape([-1, weights.shape[-1]])
      norm = np.linalg.norm(w, axis=0)
      idx = int(k * norm.shape[0])
      med = np.sort(norm, axis=0)[idx]
      mask = (norm >= med).astype(float)
      pruned_w = mask * w

      return pruned_w, mask
  elif mode == "louizos":

    def prune(weight_dict, weight_key):
      weights = weight_dict[weight_key]
      w = weights.copy()
      assert len(weights.shape) == 1
      idx = int(k * w.shape[0])
      med = np.sort(w, axis=0)[idx]
      mask = (w >= med).astype(float)
      pruned_w = mask * w

      return pruned_w, mask

  return prune


@register
def ard(k=0.5):

  def prune(weight_dict, weight_key):
    weights = weight_dict[weight_key]
    w = weights.copy()
    if len(weights.shape) == 4:
      w = w.reshape([-1, weights.shape[-1]])
    norm = np.linalg.norm(w, axis=1, keepdims=True)
    idx = int(k * norm.shape[0])
    med = np.sort(norm, axis=0)[idx]
    mask = (norm >= med).astype(float)
    pruned_w = mask * w

    return pruned_w, mask

  return prune


def prune_weights(prune_fn,
                  weights,
                  louizos_masks=None,
                  smallify_masks=None,
                  hparams=None):
  weights_pruned = {}

  pre_prune_nonzero = 0
  pre_prune_total = 0
  if louizos_masks:
    orig_weights = dict(weights)
    for weight_name in weights:
      if weight_name not in louizos_masks.keys():
        print("WARN louizos: mask not found for {}".format(weight_name))
        continue
      weights[weight_name] = louizos_masks[weight_name]
  elif smallify_masks:
    orig_weights = dict(weights)
    for weight_name in weights:
      if weight_name not in smallify_masks.keys():
        print("WARN smallify: not pruning {}".format(weight_name))
        continue
      mask = smallify_masks[weight_name]
      weights[weight_name] = weights[weight_name] * mask

  for weight_name in weights:
    if "variational" in weight_name:
      print("WARN variational: not pruning {}".format(weight_name))
      continue

    pre_prune_nonzero += np.count_nonzero(weights[weight_name])
    pre_prune_total += weights[weight_name].size

    weights_pruned[weight_name], mask = prune_fn(weights, weight_name)
    if louizos_masks or smallify_masks:
      print("applied masks to", weight_name)
      weights_pruned[weight_name] = mask * orig_weights[weight_name].reshape(
          [-1, orig_weights[weight_name].shape[-1]])

  return weights_pruned, {
      "pre_prune_nonzero": pre_prune_nonzero,
      "pre_prune_total": pre_prune_total
  }


def get_louizos_masks(sess, weights):
  masks = {}
  for weight_name in weights:
    m_name = weight_name.strip("DW") + "louizos/gates"
    m = tf.contrib.framework.get_variables_by_name(m_name)
    assert len(m) == 1
    m = m[0]
    masks[weight_name] = sess.run(m)

  return masks


def get_smallify_masks(sess, weights):
  masks = {}
  for weight_name in weights:
    switch_name = weight_name.strip("DW") + "smallify/switch"
    mask_name = weight_name.strip("DW") + "smallify/mask"
    switch = tf.contrib.framework.get_variables_by_name(switch_name)
    mask = tf.contrib.framework.get_variables_by_name(mask_name)
    assert len(switch) == 1 and len(mask) == 1
    switch, mask = switch[0], mask[0]
    switch, mask = sess.run((switch, mask))

    masks[weight_name] = switch * mask

  return masks


def is_prunable_weight(weight):
  necessary_tokens = ["kernel", "DW", "variational"]
  blacklisted_tokens = ["logit", "fc", "init", "switch", "mask", "log_sigma"]

  contains_a_necessary_token = any(t in weight.name for t in necessary_tokens)
  contains_a_blacklisted_token = any(
      t in weight.name for t in blacklisted_tokens)

  is_prunable = contains_a_necessary_token and not contains_a_blacklisted_token

  if not is_prunable:
    print("WARN: not pruning %s" % weight.name)

  return is_prunable


def get_current_weights(sess):
  weights = {}
  variables = {}
  for v in tf.trainable_variables():
    if is_prunable_weight(v):
      name = v.name.strip(":0")
      variables[name] = v

  graph = tf.get_default_graph()
  node_defs = [n for n in graph.as_graph_def().node if 'log_alpha' in n.name]

  for n in node_defs:
    weights[n.name] = sess.run(graph.get_tensor_by_name(n.name + ":0"))

  for weight_name, w in variables.items():
    weights[weight_name] = sess.run(w)

  return weights


def prune_sess_weights(sess, prune_percent, FLAGS, hparams):
  current_weights = get_current_weights(sess)
  prune_fn = get_prune_fn(FLAGS.prune)(k=prune_percent)
  current_weights_pruned = prune_weights(prune_fn, current_weights, None,
                                         hparams)

  print("there are ", len(tf.trainable_variables()), " weights")
  for v in tf.trainable_variables():
    if is_prunable_weight(v):
      assign_op = v.assign(
          np.reshape(current_weights_pruned[v.name.strip(":0")], v.shape))
      sess.run(assign_op)
