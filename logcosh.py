"""
The log cosh reconstruction loss function.
"""
import tensorflow as tf
from tensorflow.keras import backend

"""
cosh x = (e^2x+1)/(2*e^x)
softplus x = log(1+e^x)
"""
def log_cosh(y_true, y_pred):
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.cast(y_true, y_pred.dtype)
  def _logcosh(x):
    return x + tf.math.softplus(-2. * x) - tf.cast(tf.math.log(2.), x.dtype)
  return backend.mean(_logcosh(y_pred - y_true), axis=-1)