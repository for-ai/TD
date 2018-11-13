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


def bitshift_left(x, bits, dtype=tf.int32):
  y = x
  if x.dtype is not dtype:
    y = tf.cast(x, dtype)
  y = tf.bitwise.left_shift(y, tf.cast(bits, dtype))
  if x.dtype is not dtype:
    y = tf.cast(y, x.dtype)
  return y


def bitshift_right(x, bits, dtype=tf.int32):
  y = x
  if x.dtype is not dtype:
    y = tf.cast(x, dtype)
  y = tf.bitwise.right_shift(y, tf.cast(bits, dtype))
  if x.dtype is not dtype:
    y = tf.cast(y, x.dtype)
  return y


def apply_fixed_point(x, lower_bits, upper_bits, apply_to_grads=False):
  max_val = 2.**upper_bits
  shift_up = 2.**lower_bits
  shift_down = 2.**(-lower_bits)

  @tf.custom_gradient
  def fp(x):
    y = shift_down * tf.round(shift_up * x)

    def grad(dy):
      if apply_to_grads:
        dx = shift_down * tf.round(shift_up * dy)
        return tf.sign(dx) * tf.minimum(tf.abs(dx), max_val - 1)
      return dy

    return tf.sign(y) * tf.minimum(tf.abs(y), max_val - 1), grad

  return fp(x)


def apply_bitrot(inputs,
                 mantissa_bits_dropped=0,
                 exponent_shift=0,
                 exponent_safe_zone=(0, 0),
                 keep_prob=0.0,
                 is_training=False,
                 version="1",
                 apply_to_grads=False):
  """Applies bitrot to the bits of the entries of a tensor.

  Args:
    inputs: Tensor of type tf.float32, inputs to apply targeted dropout to.
    mantissa_bits_dropped: Scalar integer Tensor or python scalar (max: 23),
      sets the number of low-order mantissa bits to zero. For bfloat16, set to
      7.
    exponent_shift: Scalar integer Tensor or python scalar (max: 255), sets
      the maximum shift towards the exponent_safe_zone.
    exponent_safe_zone: Pair of scalar integer Tensors, the desired minimum and
      maximum exponent values.
    keep_prob: Scalar Tensor, passed as `tf.nn.dropout`'s `keep_prob` argument.
    is_training: bool, indicates whether currently training.

  Returns:
    Tensor, same shape and dtype as `inputs`.
  """
  is_weight = isinstance(inputs, tf.Variable)

  def bitrot_fn(x=inputs, exponent_shift=exponent_shift):
    inputs_bits = tf.bitcast(x, tf.int32)

    if isinstance(exponent_shift, tf.Tensor) or exponent_shift > 0:
      if version == "1":
        # 1s over sign and mantissa bits
        exp_mask = tf.constant(
            sum(2**i for i in range(23, 31)), dtype=tf.int32)
        mant_mask = tf.bitwise.invert(exp_mask)

        inputs_exp = tf.bitwise.bitwise_and(exp_mask, inputs_bits)
        inputs_exp = bitshift_right(inputs_exp, 23)
        zero_distance = inputs_exp - 127
        bottom, top = exponent_safe_zone

        top_ramp = tf.nn.relu(zero_distance - top)
        bottom_ramp = tf.nn.relu(-(zero_distance - bottom))
        safe_distance = tf.maximum(bottom_ramp, top_ramp)

        inputs_exp -= tf.sign(zero_distance) * tf.minimum(
            exponent_shift, safe_distance)
        inputs_exp = bitshift_left(inputs_exp, 23)
        inputs_mant = tf.bitwise.bitwise_and(inputs_bits, mant_mask)

        rotten_bits = tf.bitwise.bitwise_or(inputs_exp, inputs_mant)
      elif version == "2":
        _, top = exponent_safe_zone
        max_val = 2**top
        exponent_shift = 23. - tf.to_float(exponent_shift)
        rotten_bits = 2.**(-exponent_shift) * tf.round(2.**exponent_shift * x)
        rotten_bits = tf.sign(rotten_bits) * tf.minimum(
            tf.abs(rotten_bits), max_val - 1)
        rotten_bits = tf.bitcast(rotten_bits, tf.int32)
    else:
      rotten_bits = inputs_bits

    if isinstance(mantissa_bits_dropped,
                  tf.Tensor) or mantissa_bits_dropped > 0:
      rotten_bits = bitshift_right(rotten_bits, mantissa_bits_dropped)
      rotten_bits = bitshift_left(rotten_bits, mantissa_bits_dropped)

    rotten_inputs = tf.bitcast(rotten_bits, tf.float32)

    return tf.stop_gradient(rotten_inputs)

  @tf.custom_gradient
  def new_op(x):
    rotten_inputs = bitrot_fn(x)

    def grad(dy):
      clean_grad = dy
      rotten_grad = bitrot_fn(dy)

      if is_weight:
        tf.summary.histogram("difference", tf.abs(clean_grad - rotten_grad))
        tf.summary.scalar(
            "cosine",
            tf.reduce_sum(clean_grad * rotten_grad) /
            (tf.norm(clean_grad) * tf.norm(rotten_grad)))

      return rotten_grad if apply_to_grads else clean_grad

    return rotten_inputs, grad

  outputs = new_op(inputs)

  if not is_training:
    return outputs

  mask = tf.random_uniform(tf.shape(inputs)) < keep_prob
  outputs = tf.where(mask, inputs, outputs)

  return outputs


@register("bitrot")
def bitrot(w, params, is_training):
  bits_lost = params.bits_to_drop
  drop_rate = params.drop_rate

  return apply_bitrot(
      w,
      mantissa_bits_dropped=bits_lost,
      keep_prob=1 - drop_rate,
      is_training=is_training,
      apply_to_grads=params.apply_to_grads)


@register("ramping_bitrot")
def ramping_bitrot(w, params, is_training):
  drop_rate = params.drop_rate

  ramp = tf.to_float(tf.train.get_global_step()) / params.ramping_period
  ramp = tf.minimum(1., ramp)
  bits_lost = (params.ramp_top - params.ramp_bottom) * ramp + params.ramp_bottom
  bits_lost = tf.to_int32(tf.round(bits_lost))

  return apply_bitrot(
      w,
      mantissa_bits_dropped=bits_lost,
      keep_prob=1 - drop_rate,
      is_training=is_training,
      apply_to_grads=params.apply_to_grads)


@register("exprot")
def exprot(w, params, is_training):
  shift = params.max_shift
  safe_zone = params.safe_zone
  drop_rate = params.drop_rate

  return apply_bitrot(
      w,
      exponent_shift=shift,
      exponent_safe_zone=safe_zone,
      keep_prob=1 - drop_rate,
      is_training=is_training,
      apply_to_grads=params.apply_to_grads)


@register("ramping_exprot")
def ramping_exprot(w, params, is_training):
  safe_zone = params.safe_zone
  drop_rate = params.drop_rate

  ramp = tf.to_float(tf.train.get_global_step()) / params.ramping_period
  ramp = tf.minimum(1., ramp)
  shift = (params.ramp_top - params.ramp_bottom) * ramp + params.ramp_bottom
  shift = tf.to_int32(tf.round(shift))

  return apply_bitrot(
      w,
      exponent_shift=shift,
      exponent_safe_zone=safe_zone,
      keep_prob=1 - drop_rate,
      is_training=is_training,
      apply_to_grads=params.apply_to_grads)


@register("ramping_exprot_v2")
def ramping_exprot_v2(w, params, is_training):
  safe_zone = params.safe_zone
  drop_rate = params.drop_rate

  ramp = tf.to_float(tf.train.get_global_step()) / params.ramping_period
  ramp = tf.minimum(1., ramp)
  shift = (params.ramp_top - params.ramp_bottom) * ramp + params.ramp_bottom
  shift = tf.to_int32(tf.round(shift))

  return apply_bitrot(
      w,
      exponent_shift=shift,
      exponent_safe_zone=safe_zone,
      keep_prob=1 - drop_rate,
      is_training=is_training,
      version="2",
      apply_to_grads=params.apply_to_grads)


@register("fixed_point")
def fixed_point(w, params, is_training):
  lower_bits = float(params.lower_bits)
  upper_bits = float(params.upper_bits)

  return apply_fixed_point(
      w, lower_bits, upper_bits, apply_to_grads=params.apply_to_grads)


@register("ramping_exprot_to_fp")
def ramping_exprot_to_fp(w, params, is_training):
  cond = tf.to_float(
      tf.train.get_global_step()) < params.ramping_period + 10000
  w_rotten = ramping_exprot(w, params, is_training)
  if is_training and isinstance(w, tf.Variable):
    w_rotten = tf.cond(
        tf.logical_and(cond,
                       tf.equal(tf.mod(tf.train.get_global_step(), 10000), 0)),
        lambda: tf.assign(w, w_rotten), lambda: w_rotten)
  return tf.where(cond, w_rotten, fixed_point(w, params, is_training))


@register("ramping_exprot_to_fp_v2")
def ramping_exprot_to_fp_v2(w, params, is_training):
  cond = tf.to_float(
      tf.train.get_global_step()) < params.ramping_period + 10000
  w_rotten = tf.where(cond, ramping_exprot_v2(w, params, is_training),
                      fixed_point(w, params, is_training))
  if is_training and isinstance(w, tf.Variable):
    w_rotten = tf.cond(
        tf.logical_and(cond,
                       tf.equal(tf.mod(tf.train.get_global_step(), 10000), 0)),
        lambda: tf.assign(w, w_rotten), lambda: w_rotten)
  return w_rotten


@register("ramping_bitrot_exprot_to_fp")
def ramping_bitrot_exprot_to_fp(w, params, is_training):
  safe_zone = params.safe_zone
  drop_rate = params.drop_rate

  ramp = tf.to_float(tf.train.get_global_step()) / params.ramping_period
  ramp = tf.minimum(1., ramp)
  shift = (params.exp_ramp_top -
           params.exp_ramp_bottom) * ramp + params.exp_ramp_bottom
  shift = tf.to_int32(tf.round(shift))
  bits_lost = (params.mant_ramp_top -
               params.mant_ramp_bottom) * ramp + params.mant_ramp_bottom
  bits_lost = tf.to_int32(tf.round(bits_lost))

  cond = tf.to_float(
      tf.train.get_global_step()) < params.ramping_period + 10000
  w_rotten = apply_bitrot(
      w,
      mantissa_bits_dropped=bits_lost,
      exponent_shift=shift,
      exponent_safe_zone=safe_zone,
      keep_prob=1 - drop_rate,
      is_training=is_training)
  if is_training and isinstance(w, tf.Variable):
    w_rotten = tf.cond(
        tf.logical_and(cond,
                       tf.equal(tf.mod(tf.train.get_global_step(), 10000), 0)),
        lambda: tf.assign(w, w_rotten), lambda: w_rotten)
  return tf.where(cond, w_rotten, fixed_point(w, params, is_training))


@register("binarize")
def binarize(w, params, is_training):
  probs = tf.maximum(0., tf.minimum(1., (w + 1.) / 2.))
  mask = tf.random_uniform(tf.shape(w)) < probs
  ones = tf.ones_like(w)

  if is_training:
    binarized = tf.where(mask, ones, -ones)
    grad = tf.maximum(-1., tf.minimum(1., w))

    if isinstance(w, tf.Variable):
      deps = [tf.assign(w, grad)]
    else:
      deps = [tf.no_op()]

    with tf.control_dependencies(deps):
      binarized = grad + tf.stop_gradient(binarized - grad)
  else:
    binarized = tf.sign(w)

  return binarized


@register("targeted_weight")
def targeted_weight_dropout(w, params, is_training):
  drop_rate = params.dropout
  targ_perc = params.dropout_botk

  w_shape = w.shape
  w = tf.reshape(w, [-1, w_shape[-1]])
  norm = tf.abs(w)
  idx = tf.to_int32(targ_perc * tf.to_float(tf.shape(w)[0]))
  threshold = tf.contrib.framework.sort(norm, axis=0)[idx]
  mask = norm < threshold[None, :]

  if not is_training:
    w = (1 - mask) * w
    w = tf.reshape(w, w_shape)
    return w

  mask = tf.where(
      tf.logical_and((1. - drop_rate) < tf.random_uniform(tf.shape(w)), mask),
      tf.ones_like(w, dtype=tf.float32), tf.zeros_like(w, dtype=tf.float32))
  w = (1 - mask) * w
  w = tf.reshape(w, w_shape)
  return w


@register("targeted_weight_random")
def targeted_weight_random(w, params, is_training):
  drop_rate = params.dropout
  targ_perc = params.dropout_botk

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
  drop_rate = params.dropout
  targ_perc = 0.95 * params.dropout_botk * tf.minimum(
      1.0,
      tf.to_float(tf.train.get_global_step()) / 20000.)
  targ_perc = targ_perc + 0.05 * params.dropout_botk * tf.maximum(
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
    mask = switch < targ_perc

  mask = 1. - tf.to_float(mask)
  mask = tf.stop_gradient(mask)

  w = mask * w
  w = tf.reshape(w, w_shape)
  return w


@register("targeted_weight_piecewise")
def targeted_weight_piecewise_dropout(w, params, is_training):
  drop_rate = params.dropout

  train_percent_20k = tf.minimum(
      1.0,
      tf.to_float(tf.train.get_global_step()) / 20000.)
  train_percent_40k = tf.minimum(
      1.0, (tf.to_float(tf.train.get_global_step()) - 20000.) / 20000.)

  targ_perc = tf.cond(
      tf.less(tf.train.get_global_step(),
              20000), lambda: train_percent_20k * 0.95,
      lambda: 0.95 + train_percent_40k * (0.04 + params.td_nines))

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

  train_percent_20k = tf.minimum(
      1.0,
      tf.to_float(tf.train.get_global_step()) / 20000.)
  train_percent_40k = tf.minimum(
      1.0, (tf.to_float(tf.train.get_global_step()) - 20000.) / 20000.)

  drop_rate = params.dropout
  targ_perc = tf.cond(
      tf.less(tf.train.get_global_step(),
              20000), lambda: train_percent_20k * 0.8,
      lambda: 0.8 + train_percent_40k * (0.1 + params.td_nines))

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
  targ_perc = params.dropout_botk

  w_shape = w.shape
  w = tf.reshape(w, [-1, w_shape[-1]])
  norm = tf.abs(w)
  idx = tf.to_int32(targ_perc * tf.to_float(tf.shape(w)[0]))
  threshold = tf.contrib.framework.sort(norm, axis=0)[idx]
  mask = (norm >= threshold)[None, :]

  w = w * tf.to_float(mask)
  return tf.cond(
      tf.greater(tf.train.get_global_step(), params.dropout_delay_steps),
      lambda: tf.reshape(w, w_shape), lambda: orig_w)


@register("delayed_targeted_unit_prune")
def delayed_targeted_unit(x, params, is_training):
  orig_x = x

  w = tf.reshape(x, [-1, x.shape[-1]])
  norm = tf.norm(w, axis=0)
  idx = int(params.dropout_botk * int(w.shape[1]))
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
  return tf.nn.dropout(w, keep_prob=(1. - params.dropout))


@register("targeted_unit")
def targeted_unit_dropout(x, params, is_training):
  w = tf.reshape(x, [-1, x.shape[-1]])
  norm = tf.norm(w, axis=0)
  idx = int(params.dropout_botk * int(w.shape[1]))
  sorted_norms = tf.contrib.framework.sort(norm)
  threshold = sorted_norms[idx]
  mask = (norm < threshold)[None, :]
  mask = tf.tile(mask, [w.shape[0], 1])

  mask = tf.where(
      tf.logical_and((1. - params.dropout) < tf.random_uniform(tf.shape(w)),
                     mask), tf.ones_like(w, dtype=tf.float32),
      tf.zeros_like(w, dtype=tf.float32))
  x = tf.reshape((1 - mask) * w, x.shape)
  return x


@register("targeted_unit_random")
def targeted_unit_random(x, params, is_training):
  drop_rate = params.dropout
  targ_perc = params.dropout_botk

  w = tf.reshape(x, [-1, x.shape[-1]])

  switch = tf.Variable(
      tf.random_uniform(tf.shape(w)), name="mask", trainable=False)
  targeted_mask = tf.where((1. - targ_perc) < switch,
                           tf.ones_like(w, dtype=tf.float32),
                           tf.zeros_like(w, dtype=tf.float32))
  targeted_weights = tf.random_uniform(tf.shape(w)) * targeted_mask
  mask = tf.where((1. - drop_rate) < targeted_weights,
                  tf.ones_like(w, dtype=tf.float32),
                  tf.zeros_like(w, dtype=tf.float32))

  mask = tf.stop_gradient(mask)

  w = (1 - mask) * w
  w = tf.reshape(w, x.shape)
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
  idx = int(params.dropout_botk * int(w.shape[1]))
  sorted_norms = tf.contrib.framework.sort(norm)
  threshold = sorted_norms[idx]
  mask = (norm < threshold)[None, :, None]
  mask = tf.tile(mask, [w.shape[0], 1, w.shape[-1]])
  mask = tf.where(
      tf.logical_and((1. - params.dropout) < tf.random_uniform(tf.shape(w)),
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
      tf.random_uniform([int(w.shape[1])]) > params.dropout)[None, :]
  w = tf.reshape(mask * w, w_shape)
  return w / (1 - params.dropout)


@register("louizos_weight")
def louizos_weight_dropout(w, params, is_training):
  with tf.variable_scope("louizos"):
    EPS = 1e-8
    noise = (1 - EPS) * tf.random_uniform(w.shape) + (EPS / 2)
    rate = tf.log(1 - params.dropout) - tf.log(params.dropout)
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
    rate = tf.log(1 - params.dropout) - tf.log(params.dropout)
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
  with tf.variable_scope("smallify"):
    switch = tf.get_variable(
        "switch",
        shape=[1] * (len(x.shape) - 1) + [x.shape[-1]],
        initializer=tf.random_uniform_initializer())

    mask = tf.Variable(tf.ones_like(switch), name="mask", trainable=False)
    exp_avg = tf.Variable(tf.sign(switch), name="exp_avg", trainable=False)
    exp_std = tf.Variable(
        tf.zeros_like(switch), name="exp_std", trainable=False)
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

    mask = tf.Variable(tf.ones_like(switch), name="mask", trainable=False)
    exp_avg = tf.Variable(tf.sign(switch), name="exp_avg", trainable=False)
    exp_std = tf.Variable(
        tf.zeros_like(switch), name="exp_std", trainable=False)
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
