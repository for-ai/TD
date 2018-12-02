import tensorflow as tf

from . import defaults
from .registry import register


# MNIST =========================
@register
def mnist_basic_no_dropout():
  hps = defaults.default()
  hps.model = "basic"
  hps.data = "mnist"
  hps.activation = "relu"
  hps.batch_norm = False
  hps.drop_rate = 0.0
  hps.dropout_type = None
  hps.initializer = "glorot_uniform_initializer"
  hps.layers = [128, 64, 32]
  hps.input_shape = [784]
  hps.output_shape = [10]
  hps.layer_type = "dense"

  hps.learning_rate = 0.1
  hps.optimizer = "momentum"
  hps.momentum = 0.0

  return hps


@register
def mnist_basic_trgtd_dropout():
  hps = mnist_basic_no_dropout()
  hps.drop_rate = 0.5
  hps.dropout_type = "targeted_weight"
  hps.targ_rate = 0.5

  return hps


@register
def mnist_basic_untrgtd_dropout():
  hps = mnist_basic_no_dropout()
  hps.drop_rate = 0.25
  hps.dropout_type = "untargeted_weight"

  return hps


@register
def mnist_basic_trgtd_dropout_random():
  hps = mnist_basic_no_dropout()
  hps.drop_rate = 0.5
  hps.dropout_type = "targeted_weight_random"
  hps.targ_rate = 0.5

  return hps


@register
def mnist_basic_trgtd_unit_dropout():
  hps = mnist_basic_no_dropout()
  hps.drop_rate = 0.5
  hps.dropout_type = "targeted_unit"
  hps.targ_rate = 0.5

  return hps


@register
def mnist_basic_smallify_dropout_1eneg4():
  hps = mnist_basic_no_dropout()
  hps.dropout_type = "smallify_dropout"
  hps.smallify = 1e-4
  hps.smallify_mv = 0.9
  hps.smallify_thresh = 0.5

  return hps


@register
def mnist_basic_smallify_dropout_1eneg3():
  hps = mnist_basic_smallify_dropout_1eneg4()
  hps.smallify = 1e-3

  return hps


@register
def mnist_basic_smallify_weight_dropout_1eneg4():
  hps = mnist_basic_no_dropout()
  hps.dropout_type = "smallify_weight_dropout"
  hps.smallify = 1e-4
  hps.smallify_mv = 0.9
  hps.smallify_thresh = 0.5

  return hps


@register
def cifar10_basic_no_dropout():
  hps = defaults.default()
  hps.model = "basic"
  hps.data = "cifar10"
  hps.activation = "relu"
  hps.batch_norm = False
  hps.drop_rate = 0.0
  hps.dropout_type = None
  hps.initializer = "glorot_uniform_initializer"
  hps.layers = [128, 64, 32]
  hps.channels = 3
  hps.input_shape = [32, 32, 3]
  hps.output_shape = [10]
  hps.layer_type = "dense"

  hps.learning_rate = 0.1
  hps.optimizer = "momentum"
  hps.momentum = 0.0

  return hps


@register
def cifar100_basic_no_dropout():
  hps = cifar10_basic_no_dropout()
  hps.output_shape = [100]
  hps.data = "cifar100"
  return hps


@register
def imagenet32_basic():
  hps = defaults.default_imagenet32()
  hps.model = "basic"
  hps.activation = "relu"
  hps.batch_norm = False
  hps.drop_rate = 0.0
  hps.dropout_type = None
  hps.initializer = "glorot_uniform_initializer"
  hps.layers = [128, 64, 32]
  hps.layer_type = "dense"
  hps.learning_rate = 0.1
  hps.optimizer = "momentum"
  hps.momentum = 0.0
  return hps