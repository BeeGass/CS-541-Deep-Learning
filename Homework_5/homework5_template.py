import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#################################################################
# Insert TensorFlow code here to complete the tutorial in part 1.
#################################################################

#################################################################
# Insert TensorFlow code here to *train* the CNN for part 2.
#################################################################
yhat1 = model.predict(x_train[0:1,:,:,:])[0]  # Save model's output

#################################################################
# Write a method to extract the weights from the trained
# TensorFlow model. In particular, be *careful* of the fact that
# TensorFlow packs the convolution kernels as KxKx1xF, where
# K is the width of the filter and F is the number of filters.
#################################################################

def convertWeights (model):
    # Extract W1, b1, W2, b2, W3, b3 from model.
    # ...
    return W1, b1, W2, b2, W3, b3

#################################################################
# Below here, use numpy code ONLY (i.e., no TensorFlow) to use the
# extracted weights to replicate the output of the TensorFlow model.
#################################################################

# Implement a fully-connected layer. For simplicity, it only needs
# to work on one example at a time (i.e., does not need to be
# vectorized across multiple examples).
def fullyConnected (W, b, x):
    pass

# Implement a max-pooling layer. For simplicity, it only needs
# to work on one example at a time (i.e., does not need to be
# vectorized across multiple examples).
def maxPool (x, poolingWidth):
    pass

# Implement a softmax function.
def softmax (x):
    pass

# Implement a ReLU activation function
def relu (x):
    pass

# Load weights from TensorFlow-trained model.
W1, b1, W2, b2, W3, b3 = convertWeights(model)

# Implement the CNN with the same architecture and weights
# as the TensorFlow-trained model but using only numpy.
# yhat2 = softmax(...)

print(yhat1)
print(yhat2)
