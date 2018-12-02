import numpy as np
import tensorflow as tf

_DROPOUTS = dict()


def register(name):

  def add_to_dict(fn):
    global _DROPOUTS
    _DROPOUTS[name] = fn
    return fn

  return add_to_dict


def get_dropout(name):
  return _DROPOUTS[name]



@register("targeted_weight")
def targeted_weight_dropout(w, params, is_training):
  drop_rate = params.drop_rate
  targ_perc = params.targ_rate

  w_shape = w.shape
  w = tf.reshape(w, [-1, w_shape[-1]])
  norm = tf.abs(w)
  idx = tf.to_int32(targ_perc * tf.to_float(tf.shape(w)[0]))
  threshold = tf.contrib.framework.sort(norm, axis=0)[idx]
  mask = norm < threshold[None, :]

  if not is_training:
    w = (1. - tf.to_float(mask)) * w
    w = tf.reshape(w, w_shape)
    return w

  mask = tf.to_float(
      tf.logical_and(tf.random_uniform(tf.shape(w)) < drop_rate, mask))
  w = (1. - mask) * w
  w = tf.reshape(w, w_shape)
  return w


@register("targeted_weight_random")
def targeted_weight_random(w, params, is_training):
  drop_rate = params.drop_rate
  targ_perc = params.targ_rate

  w_shape = w.shape
  w = tf.reshape(w, [-1, w_shape[-1]])

  switch = tf.get_variable(
      "mask",
      w.shape,
      initializer=tf.random_uniform_initializer(),
      trainable=False)

  if is_training:
    mask = tf.logical_and(switch < targ_perc,
                          tf.random_uniform(w.shape) < drop_rate)
  else:
    mask = switch < targ_perc

  mask = 1. - tf.to_float(mask)
  mask = tf.stop_gradient(mask)

  w = mask * w
  w = tf.reshape(w, w_shape)
  return w


@register("ramping_targeted_weight_random")
def ramping_targeted_weight_random(w, params, is_training):
  drop_rate = params.drop_rate
  targ_perc = 0.95 * params.targ_rate * tf.minimum(
      1.0,
      tf.to_float(tf.train.get_global_step()) / 20000.)
  targ_perc = targ_perc + 0.05 * params.targ_rate * tf.maximum(
      0.0,
      tf.minimum(1.0,
                 (tf.to_float(tf.train.get_global_step()) - 20000.) / 20000.))

  w_shape = w.shape
  w = tf.reshape(w, [-1, w_shape[-1]])

  switch = tf.get_variable(
      "mask",
      w.shape,
      initializer=tf.random_uniform_initializer(),
      trainable=False)

  if is_training:
    mask = tf.logical_and(switch < targ_perc,
                          tf.random_uniform(w.shape) < drop_rate)
  else:
    mask = switch < (targ_perc * drop_rate)

  mask = 1. - tf.to_float(mask)
  mask = tf.stop_gradient(mask)

  w = mask * w
  w = tf.reshape(w, w_shape)
  return w


@register("targeted_weight_piecewise")
def targeted_weight_piecewise_dropout(w, params, is_training):
  drop_rate = params.drop_rate * tf.minimum(
      1.0,
      tf.to_float(tf.train.get_global_step()) / 40000.)

  targ_perc = 0.95 * params.targ_rate * tf.minimum(
      1.0,
      tf.to_float(tf.train.get_global_step()) / 20000.)
  targ_perc = targ_perc + 0.05 * params.targ_rate * tf.maximum(
      0.0,
      tf.minimum(1.0,
                 (tf.to_float(tf.train.get_global_step()) - 20000.) / 20000.))

  w_shape = w.shape
  w = tf.reshape(w, [-1, w_shape[-1]])
  norm = tf.abs(w)
  idx = tf.to_int32(targ_perc * tf.to_float(tf.shape(w)[0]))
  threshold = tf.contrib.framework.sort(norm, axis=0)[idx]
  mask = norm < threshold[None, :]

  if not is_training:
    w = w * (1 - tf.to_float(mask))
    return tf.reshape(w, w_shape)

  mask = tf.where(
      tf.logical_and((1. - drop_rate) < tf.random_uniform(tf.shape(w)), mask),
      tf.ones_like(w, dtype=tf.float32), tf.zeros_like(w, dtype=tf.float32))
  w = (1 - mask) * w
  w = tf.reshape(w, w_shape)
  return w


@register("targeted_unit_piecewise")
def targeted_unit_piecewise(w, params, is_training):
  drop_rate = params.drop_rate * tf.minimum(
      1.0,
      tf.to_float(tf.train.get_global_step()) / 40000.)

  targ_perc = 0.95 * params.targ_rate * tf.minimum(
      1.0,
      tf.to_float(tf.train.get_global_step()) / 20000.)
  targ_perc = targ_perc + 0.05 * params.targ_rate * tf.maximum(
      0.0,
      tf.minimum(1.0,
                 (tf.to_float(tf.train.get_global_step()) - 20000.) / 20000.))

  w_shape = w.shape
  w = tf.reshape(w, [-1, w.shape[-1]])
  norm = tf.norm(w, axis=0)
  idx = tf.to_int32(targ_perc * tf.to_float(w.shape[1]))
  sorted_norms = tf.contrib.framework.sort(norm)
  threshold = sorted_norms[idx]
  mask = (norm < threshold)[None, :]

  if not is_training:
    w = w * (1 - tf.to_float(mask))
    return tf.reshape(w, w_shape)

  mask = tf.tile(mask, [w.shape[0], 1])
  mask = tf.where(
      tf.logical_and((1. - drop_rate) < tf.random_uniform(tf.shape(w)), mask),
      tf.ones_like(w, dtype=tf.float32), tf.zeros_like(w, dtype=tf.float32))
  w = tf.reshape((1 - mask) * w, w_shape)
  return w


@register("delayed_targeted_weight_prune")
def delayed_targeted_weight(w, params, is_training):
  orig_w = w
  targ_perc = params.targ_rate

  w_shape = w.shape
  w = tf.reshape(w, [-1, w_shape[-1]])
  norm = tf.abs(w)
  idx = tf.to_int32(targ_perc * tf.to_float(tf.shape(w)[0]))
  threshold = tf.contrib.framework.sort(norm, axis=0)[idx]
  mask = norm >= threshold[None, :]

  w = w * tf.to_float(mask)
  cond = tf.to_float(tf.train.get_global_step() >= params.dropout_delay_steps)
  return cond * tf.reshape(w, w_shape) + (1 - cond) * orig_w


@register("delayed_targeted_unit_prune")
def delayed_targeted_unit(x, params, is_training):
  orig_x = x

  w = tf.reshape(x, [-1, x.shape[-1]])
  norm = tf.norm(w, axis=0)
  idx = int(params.targ_rate * int(w.shape[1]))
  sorted_norms = tf.contrib.framework.sort(norm)
  threshold = sorted_norms[idx]
  mask = (norm >= threshold)[None, None]

  w = w * tf.to_float(mask)
  return tf.cond(
      tf.greater(tf.train.get_global_step(), params.dropout_delay_steps),
      lambda: tf.reshape(w, x.shape), lambda: orig_x)


@register("untargeted_weight")
def untargeted_weight(w, params, is_training):
  if not is_training:
    return w
  return tf.nn.dropout(w, keep_prob=(1. - params.drop_rate))


@register("targeted_unit")
def targeted_unit_dropout(x, params, is_training):
  w = tf.reshape(x, [-1, x.shape[-1]])
  norm = tf.norm(w, axis=0)
  idx = int(params.targ_rate * int(w.shape[1]))
  sorted_norms = tf.contrib.framework.sort(norm)
  threshold = sorted_norms[idx]
  mask = (norm < threshold)[None, :]
  mask = tf.tile(mask, [w.shape[0], 1])

  mask = tf.where(
      tf.logical_and((1. - params.drop_rate) < tf.random_uniform(tf.shape(w)),
                     mask), tf.ones_like(w, dtype=tf.float32),
      tf.zeros_like(w, dtype=tf.float32))
  x = tf.reshape((1 - mask) * w, x.shape)
  return x


@register("targeted_unit_random")
def targeted_unit_random(w, params, is_training):
  drop_rate = params.drop_rate
  targ_perc = params.targ_rate

  w_shape = w.shape
  w = tf.reshape(w, [-1, w_shape[-1]])

  switch = tf.get_variable(
      "mask",
      w.shape[-1],
      initializer=tf.random_uniform_initializer(),
      trainable=False)

  if is_training:
    mask = tf.logical_and(switch < targ_perc,
                          tf.random_uniform(switch.shape) < drop_rate)
  else:
    mask = switch < targ_perc

  mask = 1. - tf.to_float(mask)
  mask = tf.stop_gradient(mask[None, :])

  w = mask * w
  w = tf.reshape(w, w_shape)
  return w


@register("targeted_ard")
def targeted_ard_dropout(w, x, params, is_training):
  if not is_training:
    return w
  x = tf.reshape(x, [-1, x.shape[-1]])
  activation_norms = tf.reduce_mean(tf.abs(x), axis=0)
  w_shape = w.shape
  w = tf.reshape(w, [-1, w_shape[-2], w_shape[-1]])
  norm = tf.norm(w, axis=(0, 2)) * activation_norms
  idx = int(params.targ_rate * int(w.shape[1]))
  sorted_norms = tf.contrib.framework.sort(norm)
  threshold = sorted_norms[idx]
  mask = (norm < threshold)[None, :, None]
  mask = tf.tile(mask, [w.shape[0], 1, w.shape[-1]])
  mask = tf.where(
      tf.logical_and((1. - params.drop_rate) < tf.random_uniform(tf.shape(w)),
                     mask), tf.ones_like(w, dtype=tf.float32),
      tf.zeros_like(w, dtype=tf.float32))
  w = tf.reshape((1 - mask) * w, w_shape)
  return w


@register("untargeted_unit")
def unit_dropout(w, params, is_training):
  if not is_training:
    return w
  w_shape = w.shape
  w = tf.reshape(w, [-1, w.shape[-1]])
  mask = tf.to_float(
      tf.random_uniform([int(w.shape[1])]) > params.drop_rate)[None, :]
  w = tf.reshape(mask * w, w_shape)
  return w / (1 - params.drop_rate)


@register("louizos_weight")
def louizos_weight_dropout(w, params, is_training):
  with tf.variable_scope("louizos"):
    EPS = 1e-8
    noise = (1 - EPS) * tf.random_uniform(w.shape) + (EPS / 2)
    rate = np.log(1 - params.drop_rate) - np.log(params.drop_rate)
    gates = tf.get_variable(
        "gates",
        shape=w.shape,
        initializer=tf.random_normal_initializer(mean=rate, stddev=0.01))
    if is_training:
      s = tf.nn.sigmoid(
          (gates + tf.log(noise / (1. - noise))) / params.louizos_beta)
      s_bar = s * (
          params.louizos_zeta - params.louizos_gamma) + params.louizos_gamma
    else:
      s = tf.nn.sigmoid(gates)
      s_bar = s * (
          params.louizos_zeta - params.louizos_gamma) + params.louizos_gamma
    mask = tf.minimum(1., tf.maximum(0., s_bar))

    return mask * w


@register("louizos_unit")
def louizos_unit_dropout(w, params, is_training):
  with tf.variable_scope("louizos"):
    EPS = 1e-8
    noise = (1 - EPS) * \
        tf.random_uniform([w.shape.as_list()[-1]]) + (EPS / 2)
    rate = np.log(1 - params.drop_rate) - np.log(params.drop_rate)
    gates = tf.get_variable(
        "gates",
        shape=[w.shape.as_list()[-1]],
        initializer=tf.random_normal_initializer(mean=rate, stddev=0.01))
    if is_training:
      s = tf.nn.sigmoid(
          (gates + tf.log(noise / (1. - noise))) / params.louizos_beta)
      s_bar = s * (
          params.louizos_zeta - params.louizos_gamma) + params.louizos_gamma
    else:
      s = tf.nn.sigmoid(gates)
      s_bar = s * (
          params.louizos_zeta - params.louizos_gamma) + params.louizos_gamma
    mask = tf.minimum(1., tf.maximum(0., s_bar))

    return mask * w


# from https://github.com/BayesWatch/tf-variational-dropout/blob/master/variational_dropout.py
def log_sigma2_variable(shape, ard_init=-10.):
  return tf.get_variable(
      "log_sigma2", shape=shape, initializer=tf.constant_initializer(ard_init))


# from https://github.com/BayesWatch/tf-variational-dropout/blob/master/variational_dropout.py
def get_log_alpha(log_sigma2, w):
  log_alpha = clip(log_sigma2 - paranoid_log(tf.square(w)))
  return tf.identity(log_alpha, name='log_alpha')


# from https://github.com/BayesWatch/tf-variational-dropout/blob/master/variational_dropout.py
def paranoid_log(x, eps=1e-8):
  v = tf.log(x + eps)
  return v


# from https://github.com/BayesWatch/tf-variational-dropout/blob/master/variational_dropout.py
def clip(x):
  return tf.clip_by_value(x, -8., 8.)


def dkl_qp(log_alpha):
  k1, k2, k3 = 0.63576, 1.8732, 1.48695
  C = -k1
  mdkl = k1 * tf.nn.sigmoid(k2 + k3 * log_alpha) - 0.5 * tf.log1p(
      tf.exp(-log_alpha)) + C
  return -tf.reduce_sum(mdkl)


@register("variational")
def variational_dropout(w, _, is_training):
  with tf.variable_scope("variational"):
    log_sigma2 = log_sigma2_variable(w.get_shape())
    log_alpha = get_log_alpha(log_sigma2, w)
    select_mask = tf.cast(tf.less(log_alpha, 3), tf.float32)

    if is_training:
      return w, log_alpha

    return w * select_mask, log_alpha


@register("variational_unit")
def variational_unit_dropout(w, _, is_training):
  with tf.variable_scope("variational"):
    log_sigma2 = log_sigma2_variable(int(w.shape[-1]))
    log_sigma2 = tf.reshape(log_sigma2, [1, 1, 1, -1])
    log_sigma2 = tf.tile(log_sigma2, [w.shape[0], w.shape[1], w.shape[2], 1])
    log_alpha = get_log_alpha(log_sigma2, w)
    select_mask = tf.cast(tf.less(log_alpha, 3), tf.float32)

    if is_training:
      return w, log_alpha

    return w * select_mask, log_alpha


@register("smallify_dropout")
def smallify_dropout(x, hparams, is_training):
  with tf.variable_scope("smallify", reuse=tf.AUTO_REUSE):
    switch = tf.get_variable(
        "switch",
        shape=[1] * (len(x.shape) - 1) + [x.shape[-1]],
        initializer=tf.random_uniform_initializer())

    mask = tf.get_variable(
        initializer=lambda: tf.ones_like(switch.initialized_value()),
        name="mask",
        trainable=False)
    exp_avg = tf.get_variable(
        initializer=lambda: tf.sign(switch.initialized_value()),
        name="exp_avg",
        trainable=False)
    exp_std = tf.get_variable(
        initializer=lambda: tf.zeros_like(switch.initialized_value()),
        name="exp_std",
        trainable=False)
    gates = switch * mask

    batch_sign = tf.sign(switch)
    diff = batch_sign - exp_avg

    new_mask = tf.cast(tf.less(exp_std, hparams.smallify_thresh), tf.float32)

    if not is_training:
      return tf.identity(x * gates, name="smallified")

    with tf.control_dependencies([
        tf.assign(mask, mask * new_mask),
        tf.assign(
            exp_std, hparams.smallify_mv * exp_std +
            (1 - hparams.smallify_mv) * diff**2),
        tf.assign(
            exp_avg, hparams.smallify_mv * exp_avg +
            (1 - hparams.smallify_mv) * batch_sign)
    ]):
      return tf.identity(x * gates, name="smallified")


@register("smallify_weight_dropout")
def smallify_weight_dropout(x, hparams, is_training):
  with tf.variable_scope("smallify"):
    switch = tf.get_variable(
        "switch", shape=x.shape, initializer=tf.random_uniform_initializer())

    mask = tf.get_variable(
        initializer=lambda: tf.ones_like(switch.initialized_value()),
        name="mask",
        trainable=False)
    exp_avg = tf.get_variable(
        initializer=lambda: tf.sign(switch.initialized_value()),
        name="exp_avg",
        trainable=False)
    exp_std = tf.get_variable(
        initializer=lambda: tf.zeros_like(switch.initialized_value()),
        name="exp_std",
        trainable=False)
    gates = switch * mask

    batch_sign = tf.sign(switch)
    diff = batch_sign - exp_avg

    new_mask = tf.cast(tf.less(exp_std, hparams.smallify_thresh), tf.float32)

    if not is_training:
      return tf.identity(x * gates, name="smallified")

    with tf.control_dependencies([
        tf.assign(mask, mask * new_mask),
        tf.assign(
            exp_std, hparams.smallify_mv * exp_std +
            (1 - hparams.smallify_mv) * diff**2),
        tf.assign(
            exp_avg, hparams.smallify_mv * exp_avg +
            (1 - hparams.smallify_mv) * batch_sign)
    ]):
      return tf.identity(x * gates, name="smallified")
