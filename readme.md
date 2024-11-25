DeSpaWN: Denoising Sparse Wavelet Network
The repository contains an implementation of the Denoising Sparse Wavelet Network (DeSpaWN) in TensorFlow and Keras. DeSpaWN is a deep learning architecture suitable for prospection of high-frequency time series data under an unsupervised manner. Its feature includes:
A learnable wavelet transform
Various kernel constraints (CQF, PerLayer, PerFilter Free)
Learnable hard-thresholding, used to denoise
Architecture is flexible with regards to input dimensions and decomposition level.
Installation
In order to run this code, you will need to have TensorFlow and Keras installed. You can install the necessary libraries using pip: 
"pip install tensorflow numpy pandas matplotlib"


Model Architecture
The DeSpaWN architecture consists of:
Kernel Layer: Implements learnable wavelet kernels
LowPassWave and HighPassWave Layers: Perform convolutions for decomposition
HardThresholdAssym Layer: Implements learnable denoising
LowPassTrans and HighPassTrans Layers: Perform transposed convolutions for reconstruction


Customization
You can customize the DeSpaWN model by adjusting parameters such as:
inputSize: Length of the input time series
kernelInit: Initial kernel size or values
level: Number of decomposition levels
kernelsConstraint: Type of kernel constraint (CQF, PerLayer, PerFilter, Free)