# MNIST-fashion

## MNIST-fashion.ipynb - simple Convolution network

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

あまり説明できない部分ですが、画像は 28*28 なので、畳み込みが2x2のkernelで画像の内容を認識できると思います。勉強速度を早めるのにそれぞれの層にbatchnormalizeを追加することにしました。ある程度、追加しない場合に比べて、早められました。最後に loss は分別に適する softmax を用い、optimizerはadamで（普通に一番早いと言われたり、思ったりしました）。
train dataから最後の5000個をvalidation dataにして後55000個はtrain dataにしました。練習するのにデータを正規化、左右返す、ランダムの部分の取り除きしました。

## Siamese Tensorflow.ipynb - Implement Siamese Convolution for verification then use oneshot test for classification

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
入力は二つ。その二つの差の絶対値を求めます。  
Dense 1

今回のkernelは5x5と3x3を使ってみました。論文の場合は224x224の画像は10x10,7x7のkernelを使ったので、28x28に5x5,3x3が効くかどうか試しました。また、今回の仕事は違って、「分別」の代わりに「類似」の仕事を与えるので、小さい部分を認識より、もっと一般的に大きい部分を認識したほうがいいと考えていました。
練習データの調整は正規化とaffine変換でした。affineは4つの変換があり、それらは回転、せん断、スケール、平行移動です。論文によると、それぞれの変換は0.5確率でするかしないかを判断します。affine変換は変換行列の掛け算で合成し、練習の際にデータを当てはめました。練習データの作成は同じクラスの組みは「同類」、違うクラスの組みは「異類」とされます。ランダムで練習データを一部「同類」に、一部「異類」にして練習しました。最後にoneshotテスト、oneshotとは「類似」に向かうモデルを「分別」の仕事にさせることです。そのため、各クラスから1個のデータを選んで、テストするデータに比較し、どちらにもっとも「類似」とされるのはその物のクラスをつけます。いわば k-Nearest Neightborhood みたいなアルゴリズムです。

結論：  
普通のネットワークは93.3％ぐらいの精度を果たしました。Convolution SiameseNetは「類似」の精度は92-95％(Validation Data)、一方「分別」の精度は87.78%しか果たしませんでした。なぜなら、SiameseNetは特に「分別」の仕事に向けではないと思います。間違っている点はほとんどT-shirt/pullover/shirt でした。その３つはかなり似てる点が多いため、時々分別には紛らわしいことになってしまいました。
