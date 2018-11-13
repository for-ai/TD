import tensorflow as tf

_INIT = dict()


def register(name):

  def add_to_dict(fn):
    global _INIT
    _INIT[name] = fn
    return fn

  return add_to_dict


def get_init(params):
  return _INIT[params.initializer](params)


@register("normal")
def normal(params):
  return tf.random_normal_initializer(mean=params.mean, stddev=params.sd)


@register("constant")
def constant(params):
  return tf.constant_initializer(0.1, tf.float32)


@register("uniform_unit_scaling")
def uniform_unit_scaling(params):
  return tf.uniform_unit_scaling_initializer()


@register("glorot_normal_initializer")
def glorot_normal_initializer(params):
  return tf.glorot_normal_initializer()


@register("glorot_uniform_initializer")
def glorot_uniform_initializer(params):
  return tf.glorot_uniform_initializer()


@register("variance_scaling_initializer")
def variance_scaling_initializer(params):
  return tf.variance_scaling_initializer()


class RandomUnitScaling(tf.keras.initializers.Initializer):

  def __call__(self, shape, dtype=None, partition_info=None):
    if len(shape) == 2:
      dim = (shape[0] + shape[1]) / 2.
    elif len(shape) == 4:
      dim = shape[0] * shape[1] * (shape[2] + shape[3]) / 2.

    m = tf.sqrt(3 / tf.to_float(dim))
    init = m * (2 * tf.random_uniform(shape) - 1)
    return init


class RandomHadamardConstant(tf.keras.initializers.Initializer):

  def __call__(self, shape, dtype=None, partition_info=None):
    dim = (shape[0] + shape[1]) / 2.

    flip = 2 * tf.round(tf.random_uniform(shape)) - 1
    m = tf.pow(dim, -1 / 2.)
    return m * flip


class RandomHadamardUnscaled(tf.keras.initializers.Initializer):

  def __call__(self, shape, dtype=None, partition_info=None):
    return 2 * tf.round(tf.random_uniform(shape)) - 1


class RandomWarpedUniform(tf.keras.initializers.Initializer):

  def __init__(self, k=2):
    self.k = k

  def __call__(self, shape, dtype=None, partition_info=None):
    if len(shape) == 2:
      dim = (shape[0] + shape[1]) / 2.
    elif len(shape) == 4:
      dim = shape[0] * shape[1] * (shape[2] + shape[3]) / 2.

    m = tf.sqrt(3 / tf.to_float(dim))

    eps = 1e-10
    unif = (1 - eps) * tf.random_uniform(shape) + eps / 2
    skew_unif = tf.nn.sigmoid(self.k * tf.log(unif / (1 - unif)))
    init = m * (2 * skew_unif - 1)
    return init


@register("warped_unif")
def warped_unif(params):
  return RandomWarpedUniform(params.k)


@register("unit_scaling")
def unit_scaling(params):
  return RandomUnitScaling()


@register("hadamard_constant")
def hadamard_constant(params):
  return RandomHadamardConstant()


@register("hadamard_unscaled")
def hadamard_unscaled(params):
  return RandomHadamardUnscaled()