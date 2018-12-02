import tensorflow as tf

from .registry import register
from .defaults import default, default_cifar10


# from https://github.com/tensorflow/models/blob/master/resnet/resnet_main.py
@register
def vgg16_default():
  vgg_default = default_cifar10()
  vgg_default.initializer = "glorot_uniform_initializer"
  vgg_default.model = "vgg"
  vgg_default.learning_rate = 0.01
  vgg_default.lr_scheme = "constant"
  vgg_default.weight_decay_rate = 0.0005
  vgg_default.num_classes = 10
  vgg_default.optimizer = "adam"
  vgg_default.adam_epsilon = 1e-6
  vgg_default.beta1 = 0.85
  vgg_default.beta2 = 0.997
  vgg_default.input_shape = [32, 32, 3]
  vgg_default.output_shape = [10]
  return vgg_default


@register
def cifar10_vgg16():
  hps = vgg16_default()
  hps.data = "cifar10"
  return hps


@register
def cifar100_vgg16_no_dropout():
  hps = vgg16_default()
  hps.data = "cifar100"

  hps.input_shape = [32, 32, 3]
  hps.output_shape = [100]
  hps.num_classes = 100
  hps.channels = 3
  hps.learning_rate = 0.0001
  return hps


@register
def cifar10_vgg16_no_dropout():
  hps = vgg16_default()
  hps.data = "cifar10"

  hps.input_shape = [32, 32, 3]
  hps.output_shape = [10]
  hps.num_classes = 10
  hps.channels = 3
  hps.learning_rate = 0.0001
  return hps


@register
def cifar100_vgg16_targeted_dropout():
  hps = cifar100_vgg16_no_dropout()
  hps.drop_rate = 0.5
  hps.dropout_type = "targeted_weight"
  hps.targ_rate = 0.5
  return hps


@register
def cifar100_vgg16_untargeted_dropout():
  hps = cifar100_vgg16_no_dropout()
  hps.drop_rate = 0.25
  hps.dropout_type = "untargeted_weight"
  return hps


@register
def cifar100_vgg16_untargeted_unit_dropout():
  hps = cifar100_vgg16_no_dropout()
  hps.drop_rate = 0.25
  hps.dropout_type = "untargeted_unit"
  return hps


@register
def cifar100_vgg16_targeted_unit_dropout():
  hps = cifar100_vgg16_no_dropout()
  hps.drop_rate = 0.5
  hps.dropout_type = "targeted_unit"
  hps.targ_rate = 0.5
  return hps


@register
def cifar100_vgg16_targeted_unit_dropout_botk75_66():
  hps = cifar100_vgg16_targeted_unit_dropout()
  hps.drop_rate = 0.66
  hps.targ_rate = 0.75
  return hps


@register
def cifar100_vgg16_louizos_unit():
  hps = cifar100_vgg16_no_dropout()
  hps.louizos_beta = 2. / 3.
  hps.louizos_zeta = 1.1
  hps.louizos_gamma = -0.1
  hps.louizos_cost = 0.001
  hps.dropout_type = "louizos_unit"
  hps.drop_rate = 0.25

  return hps


@register
def cifar100_vgg16_louizos_weight():
  hps = cifar100_vgg16_louizos_unit()
  hps.dropout_type = "louizos_weight"

  return hps


@register
def cifar100_vgg16_variational_unscaled():
  hps = cifar100_vgg16_no_dropout()
  hps.dropout_type = "variational"
  hps.drop_rate = 0.75
  hps.thresh = 3
  hps.var_scale = 1
  hps.weight_decay_rate = 0.0

  return hps


@register
def cifar100_vgg16_variational():
  hps = cifar100_vgg16_variational_unscaled()
  hps.var_scale = 1. / 100

  return hps


@register
def cifar100_vgg16_variational_unit_unscaled():
  hps = cifar100_vgg16_variational_unscaled()
  hps.dropout_type = "variational_unit"

  return hps


@register
def cifar100_vgg16_variational_unit():
  hps = cifar100_vgg16_variational_unit_unscaled()
  hps.var_scale = 1. / 100

  return hps


@register
def cifar100_vgg16_smallify_1eneg4():
  hps = cifar100_vgg16_no_dropout()
  hps.dropout_type = "smallify_dropout"
  hps.smallify = 1e-4
  hps.smallify_mv = 0.9
  hps.smallify_thresh = 0.5

  return hps


@register
def cifar100_vgg16_smallify_weight_1eneg5():
  hps = cifar100_vgg16_smallify_1eneg4()
  hps.dropout_type = "smallify_weight_dropout"
  hps.smallify = 1e-5

  return hps
