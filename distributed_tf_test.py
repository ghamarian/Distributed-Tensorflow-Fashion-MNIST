import tensorflow as tf
import numpy as np
import math
import os
import keras
import argparse
import random
import sys
import time
from utils import mnist_reader

parameter_servers = ["localhost:2222"]
workers = [ "localhost:2223",
          "localhost:2224",
          'localhost: 2225']
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})
# Input Flags
tf.app.flags.DEFINE_string("job_name", "", "'ps' / 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

server = tf.train.Server(cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index)

#Load data
train_images, train_labels = mnist_reader.load_mnist('data/fashion', kind='train')
test_images,test_labels = mnist_reader.load_mnist('data/fashion', kind='t10k')

#Data preprocessing
num_classes = 10
x_train = train_images.reshape([-1,28,28,1])
x_test= test_images.reshape([-1,28,28,1])
y_train = keras.utils.to_categorical(np.ravel(train_labels), num_classes)
y_test = keras.utils.to_categorical(np.ravel(test_labels), num_classes)
# Network parameters
n_hidden_1 = 256 # Units in first hidden layer
n_hidden_2 = 128 # Units in second hidden layer
n_input = 784 # Fashion MNIST data input (img shape: 28*28)
n_classes = 10 # Fashion MNIST total classes (0-9 digits)
image_size = 28
channel_size = 1
n_samples =  x_train.shape[0]
batch_size = 50
epochs = 1
num_iterations = n_samples//batch_size
test_step = 500

filter_size1 = 512
filter_size2 = 128
filter_size3 = 32

sigma = 1e-3
learning_rate = 0.01

LOG_DIR = 'projector'
print('parametres setting finished!')
def variable_summaries(var):
    '''attach a lot of summaries to a tensorboard'''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('seddev',stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram',var)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None,image_size, image_size,channel_size])
    y = tf.placeholder(tf.float32, [None, n_classes])
    tf.summary.image('input',x,10)

def conv2d(x,w):
    return tf.nn.conv2d(x,w, strides=[1,1,1,1],padding ='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1],strides = [1,2,2,1],padding ='SAME')

def avgpool2d(x):
    return tf.reduce_mean(x,[1,2])

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer,[-1,num_features])
    #now the shape should be [num_images, img_height * img_width*num_channels]
    return layer_flat, num_features

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))


def new_biases(length):
    return tf.Variable(tf.constant(0.01, shape=[length]))
print('FLAGS testing begin!')
if FLAGS.job_name == "ps":
    server.join()
    print('Job_name: ps!')
elif FLAGS.job_name == "worker":
    print('Training begin!')
    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

        # count the number of updates
        global_step = tf.get_variable(
            'global_step',
            [],
            initializer = tf.constant_initializer(0),
            trainable = False)


        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None,image_size, image_size,channel_size])
            y = tf.placeholder(tf.float32, [None, n_classes])
        tf.summary.image('input',x,10)

        #Build Model Architecure
        with tf.variable_scope('conv1'):
            #weights shape: [height, weight, channels, number of filters]
            w_conv1 = tf.Variable(new_weights([3,3,1,filter_size1]))
            b_conv1 = tf.Variable(new_biases(length = filter_size1))
            variable_summaries(w_conv1)
            variable_summaries(b_conv1)

            z1 = tf.add(conv2d(x,w_conv1),b_conv1)
            batch_mean1, batch_var1 = tf.nn.moments(z1,[0])
            scale1 = tf.Variable(tf.ones([filter_size1]))
            beta1 = tf.Variable(tf.zeros([filter_size1]))
            bn1 = tf.nn.batch_normalization(z1,batch_mean1, batch_var1, beta1, scale1, sigma)

            a1 = tf.nn.relu(bn1)
            d1 = tf.nn.dropout(a1,keep_prob = 0.8)

        with tf.variable_scope('conv2'):
            #weights shape: [height, weight, channels, number of filters]
            w_conv2 = tf.Variable(new_weights([3,3,512,filter_size2]))
            b_conv2 = tf.Variable(new_biases(length = filter_size2))
            variable_summaries(w_conv2)
            variable_summaries(b_conv2)

            z2 = tf.add(conv2d(d1,w_conv2),b_conv2)
            batch_mean2, batch_var2 = tf.nn.moments(z2,[0])
            scale2 = tf.Variable(tf.ones([filter_size2]))
            beta2 = tf.Variable(tf.zeros([filter_size2]))
            bn2 = tf.nn.batch_normalization(z2,batch_mean2, batch_var2, beta2, scale2, sigma)

            a2 = tf.nn.relu(bn2)
            p2 = avgpool2d(a2)

        # Flatten Convolution layer
        layer_flat, num_features = flatten_layer(p2)


                #Fully connected Layer
        with tf.variable_scope('full_connected'):
            w_fc = tf.Variable(new_weights([num_features, 10]))
            b_fc = tf.Variable(new_biases(length = 10))
            variable_summaries(w_fc)
            variable_summaries(b_fc)

            z4 = tf.add(tf.matmul(layer_flat,w_fc),b_fc)
            d4 = tf.nn.dropout(z4,keep_prob = 0.8)

            Y = tf.nn.softmax(d4)
        tf.summary.histogram('fc_layer',Y)

        #cross entropy
        with tf.name_scope('loss'):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits = Y, labels = y)
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        tf.summary.scalar('loss',cost)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(Y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy',accuracy)


        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()


    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),global_step = global_step,init_op = init_op)


    begin_time = time.time()
    frequency = 100
    with sv.prepare_or_wait_for_session(server.target) as sess:
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'train'), graph = tf.get_default_graph())
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'test'),graph = tf.get_default_graph())
        summary_writer = tf.summary.FileWriter(LOG_DIR,graph = tf.get_default_graph())

        count = 0
        #performe training cycles
        start_time = time.time()
        for i in range(num_iterations*epochs):
            offset = (i*batch_size)% size
            batch_x = x_train[(offset):(offset+batch_size),:]
            batch_y = y_train[offset:(offset+batch_size),:]

            summary, _,loss,step = sess.run([merged, optimizer,cross_entropy,global_step],feed_dict={x: batch_x, y: batch_y})
            train_writer.add_summary(summary,i)

            count += 1
            if count % frequency == 0 or i+1 == batch_count:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step: %d,'%(step + 1),
                     "Epoch: %2d, "%(epoch+1),
                     'Cost: %.4f,'%loss,
                     'AvgTime: %3.2fms'%float(elapsed_time*100/frequency))

        for i in range(test_step):
            if i % 10 == 0:
                summary, test_accuracy = sess.run([merged,accuracy],feed_dict={x: x_test, y: y_test})
                test_writer.add_summary(summary, i)
                print('Test accuracy at step %s: %s' % (i, test_accuracy))

    sv.stop()
    print('done')
