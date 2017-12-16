import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.gridspec as gridspec
import tensorflow as tf
from keras.datasets import fashion_mnist
from tensorflow.contrib.image import transform
from tqdm import tqdm
import helper.load_batch as lb
import helper.load_music as ld
import pygame
import argparse

def load_song(track):
    songs = ld.music_load(dir='Music_sample',length=8000)
    song_test = songs[track].reshape(-1,2)
    pygame.mixer.init(44100,-16,2)
    return pygame.sndarray.make_sound(song_test)

def load_data():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_test = [np.reshape(_,(-1,28,28,1))/255. for _ in [X_train, X_test]]
    X_val, y_val = X_train[-5000:], y_train[-5000:]
    X_train, y_train = X_train[:-5000], y_train[:-5000]
    return X_train, y_train, X_val, y_val, X_test, y_test

def prob(x):
    return tf.less(tf.random_uniform([1]),x)[0]

def rotate(x):
    theta = tf.random_uniform([1],-10,10)[0]/(180)*np.pi
    sub_tran = tf.convert_to_tensor([[tf.cos(theta),tf.sin(theta)],[-tf.sin(theta), tf.cos(theta)]])
    return tf.matmul(x, sub_tran)

def shear(x):
    p = tf.random_uniform([2],-0.1,0.1)
    sub_tran = tf.convert_to_tensor([[1,p[1]],[p[0], 1]])
    return tf.matmul(x, sub_tran)

def scale(x):
    s = tf.random_uniform([2],0.8,1.2)
    sub_tran = tf.convert_to_tensor([[s[0],0],[0,s[1]]])
    return tf.matmul(x, sub_tran)

def translation():
    return tf.random_uniform([2],-1,1)

def affine_transform(X, rate):
    trans_matrix = tf.eye(2)
    trans_matrix = tf.cond(prob(rate),lambda: rotate(trans_matrix), lambda: trans_matrix)
    trans_matrix = tf.cond(prob(rate),lambda: shear(trans_matrix), lambda: trans_matrix)
    trans_matrix = tf.cond(prob(rate),lambda: scale(trans_matrix), lambda: trans_matrix)
    t = tf.cond(prob(rate), translation, lambda: tf.zeros(2))
    a0,a1,b0,b1 = trans_matrix[0][0],trans_matrix[0][1],trans_matrix[1][0],trans_matrix[1][1]
    a2,b2 = t[0],t[1]
    return transform(X, [a0,a1,a2,b0,b1,b2,0,0])

def run(model_save=None, model_load='model/siamese2', train_loop = 1e5, reg_str=1e-4, lr_str = 1e-3, track = 0):
    sound = load_song(track)
    batch_size = 100
    val_size = 500
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    data_dict = {
        'X_train':X_train,
        'X_val':X_val,
        'X_test':X_test,
        'y_train':y_train,
        'y_val':y_val,
        'y_test':y_test,
        'num_class':10
    }
    #Initializer
    Winit = tf.random_normal_initializer(mean=0,stddev=0.01)
    binit = tf.random_normal_initializer(mean=0.5,stddev=0.01)
    Denseinit = tf.random_normal_initializer(mean=0,stddev=0.2)

    def model(X, reg_pow=0.0001):
        reg = tf.contrib.layers.l2_regularizer(scale=reg_pow)
        with tf.variable_scope('conv1'):
            X = tf.layers.conv2d(X,64,kernel_size=[5,5],strides=[1,1],activation=tf.nn.relu
                         ,kernel_initializer=Winit,bias_initializer=binit,kernel_regularizer=reg)
            X = tf.layers.max_pooling2d(X,[2,2],2)
        with tf.variable_scope('conv2'):
            X = tf.layers.conv2d(X,128,kernel_size=[3,3],strides=[1,1],activation=tf.nn.relu
                            ,kernel_initializer=Winit,bias_initializer=binit,kernel_regularizer=reg)
            X = tf.layers.max_pooling2d(X,[2,2],2)
            X = tf.contrib.layers.flatten(X)
        with tf.variable_scope('full1'):
            X = tf.layers.dense(X,1024,kernel_initializer=Denseinit,kernel_regularizer=reg,activation=tf.sigmoid)
            X = tf.layers.batch_normalization(X,training=is_training)
            return X

    def combine_predict(X1,X2):
        dim = X1.get_shape()[1]
        diff = tf.abs(X1-X2)
        out = tf.layers.dense(diff,1,kernel_initializer=Denseinit, use_bias=False)
        return tf.reduce_sum(diff,axis=1), out

    X1 = tf.placeholder(tf.float32,[None,28,28,1])
    X2 = tf.placeholder(tf.float32,[None,28,28,1])
    y = tf.placeholder(tf.int32,[None,1])
    is_training = tf.placeholder(tf.bool)

    XL,XR = tf.cond(is_training,
                    lambda:(affine_transform(X1,0.5),affine_transform(X2,0.5)),
                    lambda: (X1,X2))
    with tf.variable_scope('bottleneck') as scope:
        Y1 = model(XL, reg_str)
        scope.reuse_variables()
        Y2 = model(XR, reg_str)
    dist, p = combine_predict(Y1,Y2)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y,tf.float32),logits=p)) \
        + tf.reduce_sum(reg_losses)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_str)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(loss)
    predict = tf.cast(tf.greater(p,0.), dtype=tf.int32)
    correct_pred = tf.equal(predict, y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    sess = tf.Session()
    saver = tf.train.Saver()
    if model_load:
        try:
            saver.restore(sess,model_load)
            print('loaded successfuly')
        except:
            sess.run(tf.global_variables_initializer())
            print('Sorry... cannot load so reset the weight')

    data = lb.BatchLoader(**data_dict)
    hist = {'train':[],'val':[]}
    hist_acc = {'train':[],'val':[]}
    pbar = tqdm(total=100)
    for i in range(train_loop):
        if i % 100 == 0:
            pbar.close()
            pbar = tqdm(total=100)
        pbar.update(1)
        for phase in ['train','val']:
            if phase == 'train':
                step = [loss, accuracy, train_step]
                first, second, expect = data.make_batch(batch_size, dat_type='train')
                feed = {X1:first,X2:second,y:expect,is_training:True}
            else:
                step = [loss, accuracy, correct_pred]
                first, second, expect = data.make_batch(val_size, dat_type='val')
                feed = {X1:first,X2:second,y:expect,is_training:False}
            current_loss, acc, _ = sess.run(step, feed_dict=feed)
            hist[phase].append(current_loss)
            hist_acc[phase].append(acc)
            if i % 100 == 0:
                print('{} loss is {} and accuracy is {}'.format(phase, current_loss, acc))
    try:
        pbar.close()
    except:
        pass
    if model_save is None:
        model_save = model_load
    if model_save:
        path = saver.save(sess,model_save)
    sound.play()

    figs, axes = plt.subplots(1,2,figsize=(10,5))
    axes[0].plot(hist['train'])
    axes[0].plot(hist['val'])
    axes[0].set_title('train/val loss')
    axes[1].plot(hist_acc['train'])
    axes[1].plot(hist_acc['val'])
    axes[1].set_title('train/val acc')
    plt.save('trainig_progress.jpg')

def arg_parse():
    parser = argparse.ArgumentParser(description="Hyper parameter")
    parser.add_argument("--model_save", help="Directory to save model", \
        default="model/siamese2", type=str)
    parser.add_argument("--model_load", help="to load model Directory, if not load \
        parse blank string", default="model/siamese2", type=str)
    parser.add_argument("--train_loop", help="Looping count", default=1e5, type=int)
    parser.add_argument("--reg_str", help="regularization strenght", default=1e-4, type=float)
    parser.add_argument("--lr_str", help="learning rate", default=1e-3, type=float)
    parser.add_argument("--track", help="track number to playback", default=0, type=int)
    return parser.parse_args()
#(model_save=None, model_load='model/siamese2', train_loop = 1e5, reg_str=1e-4, lr_str = 1e-3, track = 0)
if __name__ == "__main__":
    arg = arg_parse()
    run(**vars(arg))
