import collections
import six

import tensorflow as tf


def remove_summaries():
  g = tf.get_default_graph()
  key = tf.GraphKeys.SUMMARIES
  del g.get_collection_ref(key)[:]
  assert not g.get_collection(key)


# From Tensor2Tensor
def create_host_call(model_dir):
  """Construct a host_call writing scalar summaries.
  Args:
    model_dir: String containing path to train
  Returns:
    (fn, args) Pair to be called by TPUEstimator as the host_call.
  """
  graph = tf.get_default_graph()
  summaries = graph.get_collection(tf.GraphKeys.SUMMARIES)

  gs_t = tf.reshape(tf.to_int32(tf.train.get_global_step()), [1])
  summary_kwargs = collections.OrderedDict()
  for t in summaries:
    if t.op.type not in ["ScalarSummary", "HistogramSummary"]:
      tf.logging.warn("Ignoring unsupported tf.Summary type %s" % t.op.type)
      continue

    name = t.op.name
    tensor = t.op.inputs[1]
    if t.op.type == "ScalarSummary":
      assert tensor.shape.is_compatible_with([])
      if tensor.dtype == tf.int64:
        tensor = tf.to_int32(tensor)
      summary_kwargs["ScalarSummary" + name] = tf.reshape(tensor, [1])
    elif t.op.type == "HistogramSummary":
      summary_kwargs["HistogramSummary" + name] = tf.reshape(tensor, [-1])
  # When no supported summaries are found, don't create host_call. Otherwise,
  # TPU outfeed queue would enqueue global_step while host_call doesn't dequeue
  # it, eventually causing hang.
  if not summary_kwargs:
    return None
  summary_kwargs["global_step"] = gs_t

  def host_call_fn(**kwargs):
    """Training host call. Creates summaries for training metrics.
    Args:
      **kwargs: Dict of {str: Tensor} , with `Tensor` of shape `[batch]`. Must
        contain key "global_step" with value of current global_step Tensor.
    Returns:
      List of summary ops to run on the CPU host.
    """
    gs = tf.to_int64(kwargs.pop("global_step")[0])
    with tf.contrib.summary.create_file_writer(model_dir).as_default():
      with tf.contrib.summary.always_record_summaries():
        # We need to use tf.contrib.summary in order to feed the `step`.
        for name, value in sorted(six.iteritems(kwargs)):
          if name.startswith("ScalarSummary"):
            name = name[len("ScalarSummary"):]
            tf.contrib.summary.scalar(
                name, tf.reduce_mean(tf.to_float(value)), step=gs)
          elif name.startswith("HistogramSummary"):
            name = name[len("HistogramSummary"):]
            tf.contrib.summary.histogram(name, value, step=gs)
          elif name.startswith("ImageSummary"):
            name = name[len("ImageSummary"):]
            tf.contrib.summary.image(name, value, step=gs)

        return tf.contrib.summary.all_summary_ops()

  return (host_call_fn, summary_kwargs)
