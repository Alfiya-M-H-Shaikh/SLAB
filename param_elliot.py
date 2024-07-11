"""
Parameterized Elliot Activation Function.
"""
import tensorflow as tf
import tensorflow.keras as keras

"""
Initializing parameters. We observe that a initialization from Uniform distribution 
yield a better result than normal distribution.
"""
initializer0 = keras.initializers.RandomUniform(minval = -1, maxval =1)
initializer1 = keras.initializers.RandomUniform(minval = 0.5, maxval =3)


def param_elliot_function( signal, k1, k2 ,  derivative=False ):
    """ A parameterized version of Elliot activation function """
    s = 1 # steepness
    
    abs_signal = (1 + tf.math.abs(signal * s))
    if derivative:
        return 0.5 * s / abs_signal**2
    else:
        # Return the activation signal
        return (k1*(signal * s) / abs_signal + k2)

class ParamElliotfn(keras.layers.Layer):
    def __init__(self, trainable = True):
        super(ParamElliotfn, self).__init__()
        self.k1 = self.add_weight(name='k', shape = (), initializer=initializer0, trainable=trainable)
        self.k2 = self.add_weight(name='k', shape = (), initializer=initializer0, trainable=trainable)
    def call(self, inputs):
        return param_elliot_function(inputs, self.k1, self.k2 )