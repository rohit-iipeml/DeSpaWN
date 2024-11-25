**DeSpaWN: Denoising Sparse Wavelet Network**
The repository contains an implementation of the Denoising Sparse Wavelet Network (DeSpaWN) in TensorFlow and Keras. DeSpaWN is a deep learning architecture suitable for prospection of high-frequency time series data under an unsupervised manner. Its feature includes:

1) A learnable wavelet transform
2) Various kernel constraints (CQF, PerLayer, PerFilter Free)
3) Learnable hard-thresholding, used to denoise
4) Architecture is flexible with regards to input dimensions and decomposition level.
   
**Installation**
In order to run this code, you will need to have TensorFlow and Keras installed. You can install the necessary libraries using pip: 
**pip install tensorflow numpy pandas matplotlib**

**Model Architecture**
The DeSpaWN architecture consists of:

1) Kernel Layer: Implements learnable wavelet kernels
2) LowPassWave and HighPassWave Layers: Perform convolutions for decomposition
3) HardThresholdAssym Layer: Implements learnable denoising
4) LowPassTrans and HighPassTrans Layers: Perform transposed convolutions for reconstruction


**Customization**
You can customize the DeSpaWN model by adjusting parameters such as:
1) inputSize: Length of the input time series
2) kernelInit: Initial kernel size or values
3) level: Number of decomposition levels
4) kernelsConstraint: Type of kernel constraint (CQF, PerLayer, PerFilter, Free)
