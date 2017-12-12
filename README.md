# MNIST-fashion

MNIST-fashion.ipynb - simple Convolution network

構造  
*bn = batchnormalization  
2d-convolution filter 2x2x32 stride 1  
relu  
bn  
max-pooling 2x2 stride 2  
2d-convolution filter 2x2x128　stride 1  
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
Softmax Loss

あまり説明できない部分ですが、画像は 28*28 なので、畳み込みが2x2のkernelで画像の内容を認識できると思います。勉強速度を早めるのにそれぞれの層にbatchnormalizeを追加することにしました。ある程度、追加しない場合に比べて、早められました。最後に loss は分別に適する softmax を用い、optimizerはadamで（普通に一番早い言われたり、思ったりしました）。
train dataから最後の5000個をvalidation dataにして後55000個はtrain dataにしました。練習するのにデータを正規化、左右返す、ランダムの部分の取り除きしました。

Siamese Tensorflow.ipynb - Implement Siamese Convolution for verification then use oneshot test for classification

構造 (1)
2d-convolution filter 5x5x64 stride 1  
relu  
max-pooling 2x2 stride 2  
2d-convolution filter 3x3x128　stride 1  
relu  
max-pooling 2x2 stride 2  
Flatten  
Dense 1024 relu  
bn

構造 (2)
入力は
