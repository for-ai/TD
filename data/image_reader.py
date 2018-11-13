import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from .registry import register
from .dataset_maps import get_augmentation
from .data_generators import cifar_generator, mnist_generator


@register("imagenet", None)
@register("mnist", mnist_generator.generate)
@register("cifar10", cifar_generator.generate)
@register("cifar100", cifar_generator.generate)
def image_reader(data_sources, hparams, training):
  """Input function for image data."""

  def _input_fn(params=None):
    """Input function compatible with Experiment API."""
    if params is not None and "batch_size" in params:
      hparams.batch_size = params["batch_size"]

    dataset = tf.data.TFRecordDataset(
        data_sources, num_parallel_reads=4 if training else 1)
    dataset = dataset.prefetch(5 * hparams.batch_size)

    if hparams.shuffle_data:
      dataset = dataset.shuffle(5 * hparams.batch_size)

    dataset = dataset.map(get_augmentation("load_images", hparams, training))

    if hparams.data_augmentations is not None:
      for augmentation_name in hparams.data_augmentations:
        dataset = dataset.map(
            get_augmentation(augmentation_name, hparams, training))

    dataset = dataset.map(get_augmentation("set_shapes", hparams, training))
    if hparams.data_format == "channels_first":
      dataset = dataset.map(get_augmentation("transpose", hparams, training))
    dataset = dataset.repeat().batch(hparams.batch_size)
    dataset_it = dataset.make_one_shot_iterator()

    images, labels = dataset_it.get_next()
    if params is not None and "batch_size" in params:
      images = tf.reshape(images,
                          [hparams.batch_size] + images.shape.as_list()[1:])
      labels = tf.reshape(labels,
                          [hparams.batch_size] + labels.shape.as_list()[1:])
    return {"inputs": images, "labels": labels}, labels

  return _input_fn


@register("mnist_simple", None)
def mnist_simple(data_source, params, training):
  """Input function for MNIST image data."""

  mnist = input_data.read_data_sets(data_source, one_hot=True)

  data_set = mnist.train if training else mnist.test

  def _input_fn():
    input_images = tf.constant(data_set.images)

    input_labels = tf.constant(
        data_set.labels) if not params.is_ae else tf.constant(data_set.images)

    image, label = tf.train.slice_input_producer([input_images, input_labels])

    imageBatch, labelBatch = tf.train.batch(
        [image, label], batch_size=params.batch_size)

    return {"inputs": imageBatch}, labelBatch

  return _input_fn


@register("fashion", None)
def fashion(data_source, params, training):
  """Input function for MNIST image data."""

  mnist = input_data.read_data_sets(
      data_source,
      source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/',
      one_hot=True)

  data_set = mnist.train if training else mnist.test

  def _input_fn():
    input_images = tf.constant(data_set.images)

    input_labels = tf.constant(data_set.labels)
    image, label = tf.train.slice_input_producer([input_images, input_labels])

    imageBatch, labelBatch = tf.train.batch(
        [image, label], batch_size=params.batch_size)

    return {"inputs": imageBatch}, labelBatch

  return _input_fn
