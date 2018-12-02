import tensorflow as tf

from .registry import register
from .defaults import *


# from https://github.com/tensorflow/models/blob/master/resnet/resnet_main.py
@register
def resnet_default():
  hps = default_cifar10()
  hps.model = "resnet"
  hps.residual_filters = [16, 32, 64, 128]
  hps.residual_units = [5, 5, 5]
  hps.use_bottleneck = False
  hps.batch_size = 128
  hps.learning_rate = 0.4
  hps.lr_scheme = "resnet"
  hps.weight_decay_rate = 2e-4
  hps.optimizer = "momentum"
  return hps


@register
def resnet102_imagenet224():
  hps = default_imagenet224()
  hps.model = "resnet"
  hps.residual_filters = [64, 64, 128, 256, 512]
  hps.residual_units = [3, 4, 23, 3]
  hps.use_bottleneck = True
  hps.batch_size = 128 * 8
  hps.learning_rate = 0.128 * hps.batch_size / 256.
  hps.lr_scheme = "warmup_cosine"
  hps.warmup_steps = 10000
  hps.weight_decay_rate = 1e-4
  hps.optimizer = "momentum"
  hps.use_nesterov = True
  hps.initializer = "variance_scaling_initializer"
  hps.learning_rate_cosine_cycle_steps = 120000
  hps.cosine_alpha = 0.0
  return hps


@register
def resnet102_imagenet64():
  hps = resnet102_imagenet224()
  hps.input_shape = [64, 64, 3]
  return hps


@register
def resnet50_imagenet224():
  hps = resnet102_imagenet224()
  hps.residual_units = [3, 4, 6, 3]
  return hps


@register
def resnet34_imagenet224():
  hps = resnet50_imagenet224()
  hps.use_bottleneck = False
  return hps


@register
def resnet_cifar100():
  hps = resnet_default()
  hps.num_classes = 100
  return hps


@register
def cifar10_resnet32():
  hps = resnet_default()

  return hps


@register
def cifar10_resnet32_no_dropout():
  hps = cifar10_resnet32()
  hps.drop_rate = 0.0

  return hps


@register
def cifar10_resnet32_trgtd_weight():
  hps = cifar10_resnet32_no_dropout()
  hps.drop_rate = 0.5
  hps.dropout_type = "targeted_weight"
  hps.targ_rate = 0.5

  return hps


@register
def cifar10_resnet32_weight():
  hps = cifar10_resnet32_no_dropout()
  hps.drop_rate = 0.25
  hps.dropout_type = "untargeted_weight"

  return hps


@register
def cifar10_resnet32_weight_50():
  hps = cifar10_resnet32_weight()
  hps.drop_rate = 0.50

  return hps


@register
def cifar10_resnet32_trgtd_unit():
  hps = cifar10_resnet32_no_dropout()
  hps.drop_rate = 0.5
  hps.dropout_type = "targeted_unit"
  hps.targ_rate = 0.5

  return hps


@register
def cifar10_resnet32_trgtd_ard():
  hps = cifar10_resnet32_no_dropout()
  hps.drop_rate = 0.25
  hps.dropout_type = "targeted_ard"
  hps.targ_rate = 0.5

  return hps


@register
def cifar10_resnet32_unit():
  hps = cifar10_resnet32_no_dropout()
  hps.drop_rate = 0.25
  hps.dropout_type = "untargeted_unit"

  return hps


@register
def cifar10_resnet32_unit_50():
  hps = cifar10_resnet32_unit()
  hps.drop_rate = 0.50

  return hps


@register
def cifar10_resnet32_l1_1eneg3():
  hps = cifar10_resnet32_no_dropout()
  hps.l1_norm = 0.001

  return hps


@register
def cifar10_resnet32_l1_1eneg2():
  hps = cifar10_resnet32_no_dropout()
  hps.l1_norm = 0.01

  return hps


@register
def cifar10_resnet32_l1_1eneg1():
  hps = cifar10_resnet32_no_dropout()
  hps.l1_norm = 0.1

  return hps


@register
def cifar10_resnet32_trgted_weight_l1():
  hps = cifar10_resnet32_no_dropout()
  hps.drop_rate = 0.5
  hps.dropout_type = "targeted_weight"
  hps.targ_rate = 0.5
  hps.l1_norm = 0.1

  return hps


@register
def cifar10_resnet32_targeted_unit_l1():
  hps = cifar10_resnet32_no_dropout()
  hps.drop_rate = 0.5
  hps.dropout_type = "targeted_unit"
  hps.targ_rate = 0.5
  hps.l1_norm = 0.1

  return hps


@register
def cifar10_resnet32_trgtd_unit_botk75_33():
  hps = cifar10_resnet32_no_dropout()
  hps.drop_rate = 0.33
  hps.dropout_type = "targeted_unit"
  hps.targ_rate = 0.75

  return hps


@register
def cifar10_resnet32_trgtd_unit_botk75_66():
  hps = cifar10_resnet32_no_dropout()
  hps.drop_rate = 0.66
  hps.dropout_type = "targeted_unit"
  hps.targ_rate = 0.75

  return hps


@register
def cifar10_resnet32_trgtd_weight_botk75_33():
  hps = cifar10_resnet32_no_dropout()
  hps.drop_rate = 0.33
  hps.dropout_type = "targeted_weight"
  hps.targ_rate = 0.75

  return hps


@register
def cifar10_resnet32_trgtd_weight_botk75_66():
  hps = cifar10_resnet32_no_dropout()
  hps.drop_rate = 0.66
  hps.dropout_type = "targeted_weight"
  hps.targ_rate = 0.75

  return hps


@register
def cifar10_resnet32_trgtd_unit_ramping_botk90_99():
  hps = cifar10_resnet32_no_dropout()
  hps.drop_rate = 0.99
  hps.dropout_type = "targeted_unit_piecewise"
  hps.targ_rate = 0.90

  return hps


@register
def cifar10_resnet32_trgtd_weight_ramping_botk99_99():
  hps = cifar10_resnet32_no_dropout()
  hps.drop_rate = 0.99
  hps.dropout_type = "targeted_weight_piecewise"
  hps.targ_rate = 0.99
  hps.linear_drop_rate = True

  return hps


@register
def cifar10_resnet32_louizos_weight_1en3():
  hps = cifar10_resnet32_no_dropout()
  hps.louizos_beta = 2. / 3.
  hps.louizos_zeta = 1.1
  hps.louizos_gamma = -0.1
  hps.louizos_cost = 0.001
  hps.dropout_type = "louizos_weight"
  hps.drop_rate = 0.001

  return hps


@register
def cifar10_resnet32_louizos_weight_1en1():
  hps = cifar10_resnet32_louizos_weight_1en3()
  hps.louizos_cost = 0.1
  hps.dropout_type = "louizos_weight"

  return hps


@register
def cifar10_resnet32_louizos_weight_1en2():
  hps = cifar10_resnet32_louizos_weight_1en3()
  hps.louizos_cost = 0.01

  return hps


@register
def cifar10_resnet32_louizos_weight_5en3():
  hps = cifar10_resnet32_louizos_weight_1en3()
  hps.louizos_cost = 0.005

  return hps


@register
def cifar10_resnet32_louizos_weight_1en4():
  hps = cifar10_resnet32_louizos_weight_1en3()
  hps.louizos_cost = 0.0001

  return hps


@register
def cifar10_resnet32_louizos_unit_1en3():
  hps = cifar10_resnet32_no_dropout()
  hps.louizos_beta = 2. / 3.
  hps.louizos_zeta = 1.1
  hps.louizos_gamma = -0.1
  hps.louizos_cost = 0.001
  hps.dropout_type = "louizos_unit"
  hps.drop_rate = 0.001

  return hps


@register
def cifar10_resnet32_louizos_unit_1en1():
  hps = cifar10_resnet32_louizos_unit_1en3()
  hps.louizos_cost = 0.1

  return hps


@register
def cifar10_resnet32_louizos_unit_1en2():
  hps = cifar10_resnet32_louizos_unit_1en3()
  hps.louizos_cost = 0.01

  return hps


@register
def cifar10_resnet32_louizos_unit_5en3():
  hps = cifar10_resnet32_louizos_unit_1en3()
  hps.louizos_cost = 0.005

  return hps


@register
def cifar10_resnet32_louizos_unit_1en4():
  hps = cifar10_resnet32_louizos_unit_1en3()
  hps.louizos_cost = 0.0001

  return hps


@register
def cifar10_resnet32_louizos_unit_1en5():
  hps = cifar10_resnet32_louizos_unit_1en3()
  hps.louizos_cost = 0.00001

  return hps


@register
def cifar10_resnet32_louizos_unit_1en6():
  hps = cifar10_resnet32_louizos_unit_1en3()
  hps.louizos_cost = 0.000001

  return hps


@register
def cifar10_resnet32_variational_weight():
  hps = cifar10_resnet32_no_dropout()
  hps.dropout_type = "variational"
  hps.drop_rate = 0.75
  hps.thresh = 3
  hps.var_scale = 1. / 100
  hps.weight_decay_rate = None

  return hps


@register
def cifar10_resnet32_variational_weight_unscaled():
  hps = cifar10_resnet32_no_dropout()
  hps.dropout_type = "variational"
  hps.drop_rate = 0.75
  hps.thresh = 3
  hps.var_scale = 1
  hps.weight_decay_rate = None

  return hps


@register
def cifar10_resnet32_variational_unit():
  hps = cifar10_resnet32_no_dropout()
  hps.dropout_type = "variational_unit"
  hps.drop_rate = 0.75
  hps.thresh = 3
  hps.var_scale = 1. / 100
  hps.weight_decay_rate = None

  return hps


@register
def cifar10_resnet32_variational_unit_unscaled():
  hps = cifar10_resnet32_no_dropout()
  hps.dropout_type = "variational_unit"
  hps.drop_rate = 0.75
  hps.thresh = 3
  hps.var_scale = 1
  hps.weight_decay_rate = None

  return hps


@register
def cifar10_resnet32_smallify_1eneg4():
  hps = cifar10_resnet32_no_dropout()
  hps.dropout_type = "smallify_dropout"
  hps.smallify = 1e-4
  hps.smallify_mv = 0.9
  hps.smallify_thresh = 0.5

  return hps


@register
def cifar10_resnet32_smallify_1eneg3():
  hps = cifar10_resnet32_smallify_1eneg4()
  hps.smallify = 1e-3

  return hps


@register
def cifar10_resnet32_smallify_1eneg5():
  hps = cifar10_resnet32_smallify_1eneg4()
  hps.smallify = 1e-5

  return hps


@register
def cifar10_resnet32_smallify_1eneg6():
  hps = cifar10_resnet32_smallify_1eneg4()
  hps.smallify = 1e-6

  return hps


@register
def cifar10_resnet32_smallify_weight_1eneg4():
  hps = cifar10_resnet32_no_dropout()
  hps.dropout_type = "smallify_weight_dropout"
  hps.smallify = 1e-4
  hps.smallify_mv = 0.9
  hps.smallify_thresh = 0.5

  return hps


@register
def cifar10_resnet32_smallify_weight_1eneg3():
  hps = cifar10_resnet32_smallify_weight_1eneg4()
  hps.smallify = 1e-3

  return hps


@register
def cifar10_resnet32_smallify_weight_1eneg5():
  hps = cifar10_resnet32_smallify_weight_1eneg3()
  hps.smallify = 1e-5

  return hps


@register
def cifar10_resnet32_smallify_weight_1eneg6():
  hps = cifar10_resnet32_smallify_weight_1eneg3()
  hps.smallify = 1e-6

  return hps


# ================================
