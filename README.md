# MNIST-fashion

MNIST-fashion.ipynb - simple Convolution network

構造
*bn = batchnormalization
2d-convolution filter 2x2x32 stride 1
relu
bn
max-pooling 2x2 stride 2
2d-convolution filter 2x2x128
relu
bn
max-pooling 2x2 stride 2
Flatten
Dense 128 relu
bn
Dropout
Dense 64 relu
bn
Dropout
Dense 10

Siamese Tensorflo.ipynb - Implement Siamese Convolution for verification then use oneshot test for classification
