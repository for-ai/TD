import tensorflow as tf

from .defaults import default, default_cifar10
from .registry import register

# lenet


@register
def cifar_lenet():
  hps = default_cifar10()

  hps.model = "lenet"

  hps.activation = "relu"
  hps.residual = True
  hps.initializer = "glorot_normal_initializer"
  hps.kernel_size = 5
  hps.lr_scheme = "constant"
  hps.batch_size = 128

  hps.learning_rate = 0.01
  hps.optimizer = "momentum"
  hps.momentum = 0.9
  hps.use_nesterov = True

  hps.drop_rate = 0.0
  hps.dropout_type = None
  hps.targ_rate = 0.0

  hps.axis_aligned_cost = False
  hps.clp = False
  hps.logit_squeezing = False

  return hps


@register
def cifar_lenet_no_dropout():
  hps = cifar_lenet()
  return hps


@register
def cifar_lenet_weight():
  hps = cifar_lenet_no_dropout()
  hps.dropout_type = "untargeted_weight"
  hps.drop_rate = 0.25
  return hps


@register
def cifar_lenet_trgtd_weight():
  hps = cifar_lenet_no_dropout()
  hps.drop_rate = 0.5
  hps.targ_rate = 0.5
  hps.dropout_type = "targeted_weight"
  return hps


@register
def cifar_lenet_unit():
  hps = cifar_lenet_no_dropout()
  hps.drop_rate = 0.25
  hps.dropout_type = "untargeted_unit"
  return hps


@register
def cifar_lenet_trgtd_unit():
  hps = cifar_lenet_no_dropout()
  hps.drop_rate = 0.5
  hps.targ_rate = 0.5
  hps.dropout_type = "targeted_unit"
  return hps


@register
def cifar_lenet_l1():
  hps = cifar_lenet_no_dropout()
  hps.l1_norm = 0.1
  return hps


@register
def cifar_lenet_trgtd_weight_l1():
  hps = cifar_lenet_no_dropout()
  hps.l1_norm = 0.1
  hps.drop_rate = 0.5
  hps.targ_rate = 0.5
  hps.dropout_type = "targeted_weight"
  return hps


@register
def cifar_lenet_trgtd_unit_l1():
  hps = cifar_lenet_no_dropout()
  hps.l1_norm = 0.1
  hps.drop_rate = 0.5
  hps.targ_rate = 0.5
  hps.dropout_type = "targeted_unit"
  return hps


@register
def cifar_lenet_trgtd_unit_botk75_33():
  hps = cifar_lenet_no_dropout()
  hps.drop_rate = 0.33
  hps.dropout_type = "targeted_unit"
  hps.targ_rate = 0.75
  return hps


@register
def cifar_lenet_trgtd_unit_botk75_66():
  hps = cifar_lenet_no_dropout()
  hps.drop_rate = 0.66
  hps.dropout_type = "targeted_unit"
  hps.targ_rate = 0.75
  return hps


@register
def cifar_lenet_trgtd_weight_botk75_33():
  hps = cifar_lenet_no_dropout()
  hps.drop_rate = 0.33
  hps.dropout_type = "targeted_weight"
  hps.targ_rate = 0.75
  return hps


@register
def cifar_lenet_trgtd_weight_botk75_66():
  hps = cifar_lenet_no_dropout()
  hps.drop_rate = 0.66
  hps.dropout_type = "targeted_weight"
  hps.targ_rate = 0.75
  return hps


@register
def cifar_lenet_louizos_weight_1en3():
  hps = cifar_lenet_no_dropout()
  hps.louizos_beta = 2. / 3.
  hps.louizos_zeta = 1.1
  hps.louizos_gamma = -0.1
  hps.louizos_cost = 0.001
  hps.dropout_type = "louizos_weight"
  hps.drop_rate = 0.25
  return hps


@register
def cifar_lenet_louizos_weight_1en1():
  hps = cifar_lenet_no_dropout()
  hps.louizos_beta = 2. / 3.
  hps.louizos_zeta = 1.1
  hps.louizos_gamma = -0.1
  hps.louizos_cost = 0.1
  hps.dropout_type = "louizos_weight"
  hps.drop_rate = 0.25
  return hps


@register
def cifar_lenet_louizos_weight_1en2():
  hps = cifar_lenet_no_dropout()
  hps.louizos_beta = 2. / 3.
  hps.louizos_zeta = 1.1
  hps.louizos_gamma = -0.1
  hps.louizos_cost = 0.01
  hps.dropout_type = "louizos_weight"
  hps.drop_rate = 0.25
  return hps


@register
def cifar_lenet_louizos_weight_5en3():
  hps = cifar_lenet_no_dropout()
  hps.louizos_beta = 2. / 3.
  hps.louizos_zeta = 1.1
  hps.louizos_gamma = -0.1
  hps.louizos_cost = 0.005
  hps.dropout_type = "louizos_weight"
  hps.drop_rate = 0.25
  return hps


@register
def cifar_lenet_louizos_weight_1en4():
  hps = cifar_lenet_no_dropout()
  hps.louizos_beta = 2. / 3.
  hps.louizos_zeta = 1.1
  hps.louizos_gamma = -0.1
  hps.louizos_cost = 0.0001
  hps.dropout_type = "louizos_weight"
  hps.drop_rate = 0.25
  return hps


@register
def cifar_lenet_louizos_unit_1en3():
  hps = cifar_lenet_no_dropout()
  hps.louizos_beta = 2. / 3.
  hps.louizos_zeta = 1.1
  hps.louizos_gamma = -0.1
  hps.louizos_cost = 0.001
  hps.dropout_type = "louizos_unit"
  hps.drop_rate = 0.25
  return hps


@register
def cifar_lenet_louizos_unit_1en1():
  hps = cifar_lenet_no_dropout()
  hps.louizos_beta = 2. / 3.
  hps.louizos_zeta = 1.1
  hps.louizos_gamma = -0.1
  hps.louizos_cost = 0.1
  hps.dropout_type = "louizos_unit"
  hps.drop_rate = 0.25
  return hps


@register
def cifar_lenet_louizos_unit_1en2():
  hps = cifar_lenet_no_dropout()
  hps.louizos_beta = 2. / 3.
  hps.louizos_zeta = 1.1
  hps.louizos_gamma = -0.1
  hps.louizos_cost = 0.01
  hps.dropout_type = "louizos_unit"
  hps.drop_rate = 0.25
  return hps


@register
def cifar_lenet_louizos_unit_5en3():
  hps = cifar_lenet_no_dropout()
  hps.louizos_beta = 2. / 3.
  hps.louizos_zeta = 1.1
  hps.louizos_gamma = -0.1
  hps.louizos_cost = 0.005
  hps.dropout_type = "louizos_unit"
  hps.drop_rate = 0.25
  return hps


@register
def cifar_lenet_louizos_unit_1en4():
  hps = cifar_lenet_no_dropout()
  hps.louizos_beta = 2. / 3.
  hps.louizos_zeta = 1.1
  hps.louizos_gamma = -0.1
  hps.louizos_cost = 0.0001
  hps.dropout_type = "louizos_unit"
  hps.drop_rate = 0.25
  return hps


@register
def cifar_lenet_variational():
  hps = cifar_lenet_no_dropout()
  hps.dropout_type = "variational"
  hps.var_scale = 1. / 100
  hps.drop_rate = 0.75

  return hps


@register
def cifar_lenet_variational_unscaled():
  hps = cifar_lenet_no_dropout()
  hps.dropout_type = "variational"
  hps.drop_rate = 0.75

  return hps


@register
def cifar_lenet_variational_unit():
  hps = cifar_lenet_no_dropout()
  hps.dropout_type = "variational_unit"
  hps.var_scale = 1. / 100
  hps.drop_rate = 0.75

  return hps


@register
def cifar_lenet_variational_unit_unscaled():
  hps = cifar_lenet_no_dropout()
  hps.dropout_type = "variational_unit"
  hps.drop_rate = 0.75

  return hps


@register
def cifar_lenet_smallify_neg4():
  hps = cifar_lenet_no_dropout()
  hps.dropout_type = "smallify_dropout"
  hps.smallify = 1e-4
  hps.smallify_mv = 0.9
  hps.smallify_thresh = 0.5
  hps.smallify_delay = 10000
  return hps
