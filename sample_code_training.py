from despawn import createDeSpaWN, Kernel, LowPassWave, HighPassWave, LowPassTrans, HighPassTrans, HardThresholdAssym
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

# Define model parameters
input_size = 160000  # Set this to your input size, e.g., 160000 for 10s of audio at 16kHz
num_levels = 17  # Set this to log2(input_size), e.g., 17 for input_size of 160000
kernel_size = 8  # Size of the wavelet kernel, 8 is default for Daubechies-4 like wavelets
num_epochs = 1000  # Number of training epochs

# Create the DeSpaWN model
model1, model2 = createDeSpaWN(inputSize=input_size, 
                               kernelInit=kernel_size, 
                               kernTrainable=True,  # Allow kernel to be trained
                               level=num_levels, 
                               lossCoeff='l1',  # Use L1 norm for coefficient loss
                               kernelsConstraint='CQF',  # Use Conjugate Quadrature Filters constraint
                               initHT=1.0,  # Initial value for hard thresholding
                               trainHT=True)  # Allow hard thresholding to be trained

# Define loss functions
def reconstruction_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def coefficient_loss(y_true, y_pred):
    return y_pred  # The sparsity loss is already computed in the model

# Compile the model
optimizer = keras.optimizers.Nadam(learning_rate=0.001)
model1.compile(optimizer=optimizer, loss=[reconstruction_loss, coefficient_loss])

# Load and preprocess your data
# Replace this with your actual data loading code
your_data = np.random.randn(100, input_size, 1, 1)  # Example: 100 samples of input_size length

# Train the model
history = model1.fit(your_data, 
                     [your_data, np.empty((your_data.shape[0]))],  # Target for coefficient loss doesn't matter
                     epochs=num_epochs, 
                     verbose=1)

# After training, you can use model2 for inference
# Example:
test_signal = np.random.randn(1, input_size, 1, 1)
reconstructed, coefficients = model2.predict(test_signal)