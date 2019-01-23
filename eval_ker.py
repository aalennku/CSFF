import h5py
import scipy.io as sio
import tensorflow as tf
import numpy as np
import math
import time
import os

from tqdm import tqdm as tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="0"

I_VAL = 1

DATA_PATH = './Pavia_University/PaviaU.mat'
DATA_GT_PATH = './Pavia_University/PaviaU_gt.mat'

data_mat = sio.loadmat(DATA_PATH)['paviaU']#[:,:,:100]
data_mat_gt = sio.loadmat(DATA_GT_PATH)['paviaU_gt']
data_mat = data_mat.astype(np.float)

height, width, channel = data_mat.shape
shape = data_mat_gt.shape
data_mean, data_std = np.average(data_mat), np.var(data_mat)**0.5
data_mat = (data_mat - data_mean) / data_std

fea1 = 10
fea2 = fea1
fea3 = int(2*fea2)
fea4 = fea3
fea5 = int(2*fea4)
fea6 = fea5
fea7 = int(2*fea6)
fea8 = fea7

sess = tf.InteractiveSession()

classes = 2
deepth = channel

images_placeholder = tf.placeholder(tf.float32, shape=(None, 2, deepth, 1))
label_placeholder = tf.placeholder(tf.int64, shape=(None, 1))

def loss(logpros, labels):
    labels = tf.reshape(labels, [-1])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logpros, labels=labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.add_to_collection('losses', loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op


def _variable_on_gpu(name, shape, initializer):
    with tf.device('/gpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd=0.0005):
    var = _variable_on_gpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var



def conv_relu(input, kernel, bias, stride, padding):
    conv = tf.nn.conv2d(input, kernel, stride, padding=padding)
    return tf.nn.relu(conv+bias)


def inference(images):
    with tf.variable_scope('conv1') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,8,1,fea1], stddev=math.sqrt(1.0/8))
        biases = _variable_on_gpu('biases', [fea1], tf.constant_initializer(0.0))
        conv1 = conv_relu(images, weights, biases, [1,1,1,1], 'VALID')
    with tf.variable_scope('conv1_') as scope:
        weights = _variable_with_weight_decay('weights', shape=[2,1,fea1,fea1], stddev=math.sqrt(1.0/fea1))
        biases = _variable_on_gpu('biases', [fea1], tf.constant_initializer(0.0))
        conv1_ = conv_relu(conv1, weights, biases, [1,1,1,1], 'VALID')
    with tf.variable_scope('conv2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,3,fea1,fea2], stddev=math.sqrt(2.0/3/fea1))
        biases = _variable_on_gpu('biases', [fea2], tf.constant_initializer(0.0))
        conv2 = conv_relu(conv1_, weights, biases, [1,1,1,1], 'SAME')
    pool1 = tf.nn.max_pool(conv2, [1,1,3,1], [1,1,3,1], padding='SAME')
    # drop1 = tf.nn.dropout(pool1, keep_prob)

    with tf.variable_scope('conv3') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,3,fea2,fea3], stddev=math.sqrt(2.0/3/fea2))
        biases = _variable_on_gpu('biases', [fea3], tf.constant_initializer(0.0))
        conv3 = conv_relu(pool1, weights, biases, [1,1,1,1], 'VALID')
    # drop_conv3 = tf.nn.dropout(conv3, keep_prob, noise_shape=[batchsize, 1, 1, fea3])
    with tf.variable_scope('conv4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,3,fea3,fea4], stddev=math.sqrt(2.0/3/fea3))
        biases = _variable_on_gpu('biases', [fea4], tf.constant_initializer(0.0))
        conv4 = conv_relu(conv3, weights, biases, [1,1,1,1], 'VALID')
    pool2 = tf.nn.max_pool(conv4, [1,1,2,1], [1,1,2,1], padding='SAME')
    # drop_pool2 = tf.nn.dropout(pool2, keep_prob, noise_shape=[batchsize, 1, 1, fea4])

    with tf.variable_scope('conv5') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,3,fea4,fea5], stddev=math.sqrt(2.0/3/fea4))
        biases = _variable_on_gpu('biases', [fea5], tf.constant_initializer(0.0))
        conv5 = conv_relu(pool2, weights, biases, [1,1,1,1], 'VALID')
    # drop_conv5 = tf.nn.dropout(conv5, keep_prob, noise_shape=[batchsize, 1, 1, fea5])
    with tf.variable_scope('conv6') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,3,fea5,fea6], stddev=math.sqrt(2.0/3/fea5))
        biases = _variable_on_gpu('biases', [fea6], tf.constant_initializer(0.0))
        conv6 = conv_relu(conv5, weights, biases, [1,1,1,1], 'VALID')
    pool3 = tf.nn.max_pool(conv6, [1,1,2,1], [1,1,2,1], padding='SAME')
    # drop_pool3 = tf.nn.dropout(pool3, keep_prob, noise_shape=[batchsize, 1, 1, fea6])

    with tf.variable_scope('fc_conv1') as scope:
        dims = 5 #should be corrected according to paritcular dataset
        weights = _variable_with_weight_decay('weights', shape=[1,dims,fea6,fea7], stddev=math.sqrt(2.0/dims/fea6))
        biases = _variable_on_gpu('biases', [fea7], tf.constant_initializer(0.0))
        # scores = tf.nn.conv2d(pool3, weights, [1,1,1,1], padding='VALID')+biases
        fc_conv1 = conv_relu(pool3, weights, biases, [1,1,1,1], 'VALID')
        # fc_conv1_drop = tf.nn.dropout(fc_conv1, keep_prob)
    with tf.variable_scope('fc_conv2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,1,fea7,fea8], stddev=math.sqrt(2.0/fea7))
        biases = _variable_on_gpu('biases', [fea8], tf.constant_initializer(0.0))
        fc_conv2 = conv_relu(fc_conv1, weights, biases, [1,1,1,1], 'VALID')
    #     # fc_conv2_drop = tf.nn.dropout(fc_conv2, keep_prob)
    with tf.variable_scope('scores') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,1,fea8,classes], stddev=math.sqrt(1.0/fea8))
        biases = _variable_on_gpu('biases', [classes], tf.constant_initializer(0.0))
        scores = tf.nn.conv2d(fc_conv2, weights, [1,1,1,1], padding='VALID') + biases
    logits_flat = tf.reshape(scores, [-1, classes])

    return logits_flat

# load from check point
def load_ckpt():
    # global_step = tf.Variable(0, trainable=False)
    # saver = tf.train.Saver(tf.all_variables())
    with tf.variable_scope('inference') as scope:
        logits = inference(images_placeholder)
    loss_ = loss(logits, label_placeholder)

    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('checkpoint/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found!')
        return

load_ckpt()

def test(images):
    prediction_list = []
    with tf.variable_scope('inference', reuse=True) as scope:
        logpros = inference(images_placeholder)
        y_conv = tf.nn.softmax(logpros)
    predictions = sess.run(y_conv, feed_dict={images_placeholder: images})
    prediction_list.extend(predictions)
    return prediction_list


time_list = []



kernel = 10
flag = False
import sys

f = open('ker_%d.txt'%(I_VAL),'w') 

to_test = np.array([]).reshape((-1,2,deepth,1))

total_number = np.sum(data_mat_gt>0)
cnt = 0
for idx_i in tqdm(xrange(data_mat.shape[0])):
    for idx_j in tqdm(xrange(data_mat.shape[1]),leave=False):
        if data_mat_gt[idx_i,idx_j] == 0:
            continue

        cnt += 1
        flag = True
        patch_data = data_mat[max(idx_i-kernel+1,0):min(idx_i+kernel,shape[0]),\
                   max(idx_j-kernel+1,0):min(idx_j+kernel,shape[1]),:]
        patch_shape = patch_data.shape[:2]
        f.write('_(%d,%d)_(%d,%d)\n'%(idx_i,idx_j,patch_shape[0],patch_shape[1]))
        patch_data = patch_data.reshape((-1,1,deepth))
#       print patch_data.shape
        center_pixel = (data_mat[idx_i,idx_j,:].reshape(1,1,-1)).repeat(patch_data.shape[0],axis=0)
#       print center_pixel.shape
        patch_data_double = np.concatenate((patch_data, center_pixel), axis=1)
        patch_data_double = patch_data_double.reshape(-1,2,deepth,1)
            
        to_test = np.concatenate((to_test,patch_data_double),axis=0)
            
        if to_test.shape[0] > 77777:
            patch_labels = test(to_test)
            labels = np.array(patch_labels)
            labels = labels[:,1]
            to_test = np.array([]).reshape((-1,2,deepth,1))
            f.write(str(labels.tolist())+'\n')
if to_test.shape[0] != 0:
    patch_labels = test(to_test)
    labels = np.array(patch_labels)
    labels = labels[:,1]
    to_test = []
    f.write(str(labels.tolist())+'\n')
f.close()

