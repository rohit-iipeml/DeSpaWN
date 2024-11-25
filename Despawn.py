import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#Kernel Layer: Implements learnable wavelet kernels
class Kernel(tf.keras.layers.Layer):
    def __init__(self, kernelInit=8, trainKern=True, **kwargs):
        self.trainKern  = trainKern
        if isinstance(kernelInit,int):
            self.kernelSize = kernelInit
            self.kernelInit = 'random_normal'
        else:
            self.kernelSize = kernelInit.__len__()
            self.kernelInit = tf.constant_initializer(kernelInit)
        super(Kernel, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(shape       = (self.kernelSize,1,1,1),
                                      initializer = self.kernelInit,
                                      trainable   = self.trainKern, name='kernel')
        super(Kernel, self).build(input_shape)
    def call(self, inputs):
        return self.kernel

# LowPassWave and HighPassWave Layers: Perform convolutions for decomposition
#Low Pass Wavelet Layer
#performs a convolution between the signal(inputs[0]) and the wavelet kernel (inputs[1]) with a stride of (2,1) ---------> "stride" refers to the number of pixels by which a filter moves across the input image during the convolution operation
#This will reduce the signal size and focus on low frequency components (smooth trends).

class LowPassWave(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LowPassWave, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LowPassWave, self).build(input_shape)

    def call(self, inputs):
        return tf.nn.conv2d(inputs[0], inputs[1], padding="SAME", strides=(2, 1))


# Layer that performs a convolution between its two inputs with stride (2,1).
# Performs first the reverse alternative flip on the second inputs
class HighPassWave(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(HighPassWave, self).__init__(**kwargs)

    def build(self, input_shape):
        self.qmfFlip = tf.reshape(tf.Variable([(-1)**(i) for i in range(input_shape[1][0])],
                                              dtype='float32', name='mask', trainable=False),(-1,1,1,1))
        super(HighPassWave, self).build(input_shape)

    def call(self, inputs):
        return tf.nn.conv2d(inputs[0], tf.math.multiply(tf.reverse(inputs[1],[0]),self.qmfFlip),
                            padding="SAME", strides=(2, 1))

# HardThresholdAssym Layer: Implements learnable denoising
#Hard Threshold Layer for denoising
#Applied a thresholding operation to suppress noise while retaining important components
#we will add learnable thresholds
#For values greater than thrP, the output will be preserved
#For values less tham thrN, the output will be suppressed

class HardThresholdAssym(tf.keras.layers.Layer):
    def __init__(self, init=1.0, trainBias=True, **kwargs):
        self.init = tf.constant_initializer(init) if isinstance(init, (float, int)) else 'ones'
        self.trainBias = trainBias
        super(HardThresholdAssym, self).__init__(**kwargs)

    def build(self, input_shape):
        self.thrP = self.add_weight(shape=(1, 1, 1, 1), initializer=self.init,
                                    trainable=self.trainBias, name='threshold_pos')
        self.thrN = self.add_weight(shape=(1, 1, 1, 1), initializer=self.init,
                                    trainable=self.trainBias, name='threshold_neg')
        super(HardThresholdAssym, self).build(input_shape)

    def call(self, inputs):
        return tf.math.multiply(inputs, tf.math.sigmoid(10*(inputs-self.thrP)) +
                                tf.math.sigmoid(-10*(inputs+self.thrN)))

#LowPassTrans and HighPassTrans Layers: Perform transposed convolutions for reconstruction
# Layer that performs a convolution transpose between its two inputs with stride (2,1).
# The third input specifies the size of the reconstructed signal (to make sure it matches the decomposed one)
class LowPassTrans(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LowPassTrans, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LowPassTrans, self).build(input_shape)

    def call(self, inputs):
        return tf.nn.conv2d_transpose(inputs[0], inputs[1], inputs[2], padding="SAME", strides=(2, 1))
    
# Layer that performs a convolution transpose between its two inputs with stride (2,1).
# Performs first the reverse alternative flip on the second inputs
# The third input specifies the size of the reconstructed signal (to make sure it matches the decomposed one)
class HighPassTrans(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(HighPassTrans, self).__init__(**kwargs)

    def build(self, input_shape):
        self.qmfFlip     = tf.reshape(tf.Variable([(-1)**(i) for i in range(input_shape[1][0])],
                                                  dtype='float32', name='mask', trainable=False),(-1,1,1,1))
        super(HighPassTrans, self).build(input_shape)

    def call(self, inputs):
        return tf.nn.conv2d_transpose(inputs[0], tf.math.multiply(tf.reverse(inputs[1],[0]),self.qmfFlip),
                                      inputs[2], padding="SAME", strides=(2, 1))


"""
    Function that generates a TF DeSpaWN network
    Parameters
    ----------
    inputSize : INT, optional
        Length of the time series. Network is more efficient if set.
        Can be set to None to allow various input size time series.
        The default is None. 
    kernelInit : numpy array or LIST or INT, optional
        Initialisation of the kernel. If INT, random normal initialisation of size kernelInit.
        If array or LIST, then kernelInit is the kernel.
        The default is 8.
    kernTrainable : BOOL, optional
        Whether the kernels are trainable. Set to FALSE to compare to traditional wavelet decomposition. 
        The default is True.
    level : INT, optional
        Number of layers in the network.
        Ideally should be log2 of the time series length.
        If bigger, additional layers will be of size 1.
        The default is 1.
    lossCoeff : STRING, optional
        To specify which loss on the wavelet coefficient to compute.
        Can be None (no loss computed) or 'l1'' for the L1-norm of the coefficients.
        The default is 'l1'.
    kernelsConstraint : STRING, optional
        Specify which version of DeSpaWN to implement.
        The default is 'CQF'.
    initHT : FLOAT, optional
        Value to initialise the Hard-thresholding coefficient.
        The default is 1.0.
    trainHT : BOOL, optional
        Whether the hard-thresholding coefficient is trainable or not.
        Set to FALSE to compare to traditiona wavelet decomposition.
        The default is True.

    Returns
    -------
    model1: a TF neural network with outputs the reconstructed signals and the loss on the wavelet coefficients
    model2: a TF neural network with outputs t the reconstructed signals and wavelet coefficients

    model1 and model2 share their architecture, weigths and parameters.
    Training one of the two changes both models
    """
class ShapeLayer(keras.layers.Layer):
    def call(self, inputs):
        return tf.shape(inputs)
class LossCoeffLayer(keras.layers.Layer):
    def call(self, inputs):
        gint, hl = inputs[0], inputs[1:]
        concat = keras.ops.concatenate([gint] + hl, axis=1)
        return keras.ops.mean(keras.ops.abs(concat), axis=1, keepdims=True)
def createDeSpaWN(inputSize=None, kernelInit=8, kernTrainable=True, level=1, lossCoeff='l1', kernelsConstraint='QMF', initHT=1.0, trainHT=True):
    input_shape = (inputSize,1,1)
    inputSig = keras.layers.Input(shape=input_shape, name='input_Raw')
    g = inputSig
    if kernelsConstraint=='CQF':
        kern = Kernel(kernelInit, trainKern=kernTrainable)(g)
        kernelsG  = [kern for lev in range(level)]
        kernelsH  = kernelsG
        kernelsGT = kernelsG
        kernelsHT = kernelsG
    elif kernelsConstraint=='PerLayer':
        kernelsG  = [Kernel(kernelInit, trainKern=kernTrainable)(g) for lev in range(level)]
        kernelsH  = kernelsG
        kernelsGT = kernelsG
        kernelsHT = kernelsG
    elif kernelsConstraint=='PerFilter':
        kernelsG  = [Kernel(kernelInit, trainKern=kernTrainable)(g) for lev in range(level)]
        kernelsH  = [Kernel(kernelInit, trainKern=kernTrainable)(g) for lev in range(level)]
        kernelsGT = kernelsG
        kernelsHT = kernelsH
    elif kernelsConstraint=='Free':
        kernelsG  = [Kernel(kernelInit, trainKern=kernTrainable)(g) for lev in range(level)]
        kernelsH  = [Kernel(kernelInit, trainKern=kernTrainable)(g) for lev in range(level)]
        kernelsGT = [Kernel(kernelInit, trainKern=kernTrainable)(g) for lev in range(level)]
        kernelsHT = [Kernel(kernelInit, trainKern=kernTrainable)(g) for lev in range(level)]
    hl = []
    inSizel = []
    # Decomposition
    shape_layer = ShapeLayer()
    for lev in range(level):
        inSizel.append(shape_layer(g))
        hl.append(HardThresholdAssym(init=initHT,trainBias=trainHT)(HighPassWave()([g,kernelsH[lev]])))
        g  = LowPassWave()([g,kernelsG[lev]])
    g = HardThresholdAssym(init=initHT,trainBias=trainHT)(g)
    # save intermediate coefficients to output them
    gint = g
    # Reconstruction
    for lev in range(level-1,-1,-1):
        h = HighPassTrans()([hl[lev],kernelsHT[lev],inSizel[lev]])
        g = LowPassTrans()([g,kernelsGT[lev],inSizel[lev]])
        g = keras.layers.Add()([g,h])
    
    # Compute specified loss on coefficients
    if not lossCoeff:
        vLossCoeff = keras.ops.zeros((1, 1, 1, 1))
    elif lossCoeff == 'l1':
        vLossCoeff = LossCoeffLayer()([gint] + hl)
    else:
        raise ValueError('Could not understand value in \'lossCoeff\'. It should be either \'l1\' or None')
    return keras.models.Model(inputSig,[g,vLossCoeff]), keras.models.Model(inputSig,[g,gint,hl[::-1]])
