import pickle
import time
import json

import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Masking, LSTM
from keras import regularizers
import numpy as np


max_word = 15
batch_size = 100
state_size = 25
n_classes = 2
learning_rate = 0.1
hidden_layer1 = 500
hidden_layer2 = 300

with open('/home/janmejaya/sentiment_files/data/movie_vocab.pkl', 'rb') as f:
    data = pickle.load(f)

vocab_len = len(data['vocab_frequency_tuple'])
print('Vocab len: ', vocab_len)
vocab_len = 10000
# Weights
w_nn1 = tf.Variable(tf.random_normal([vocab_len, hidden_layer1]), name='w_nn1')
b_nn1 = tf.Variable(tf.random_normal([hidden_layer1]), name='b_nn1')
w_nn2 = tf.Variable(tf.random_normal([hidden_layer1, hidden_layer2]), name='w_nn2')
b_nn2 = tf.Variable(tf.random_normal([hidden_layer2]), name='b_nn1')
w_out = tf.Variable(tf.random_normal([hidden_layer2, 1]), name='w_out')
b_out = tf.Variable(tf.random_normal([1]), name='b_out')

x = tf.placeholder(tf.float32, [max_word, vocab_len], name='input_placeholder')
y = tf.placeholder(tf.float32, [max_word, 1], name='labels_placeholder')

## Build model

in_layer = tf.sigmoid(tf.add(tf.matmul(x, w_nn1), b_nn1))
h_layer = tf.sigmoid(tf.add(tf.matmul(in_layer, w_nn2), b_nn2))
out = tf.sigmoid(tf.add(tf.matmul(h_layer, w_out), b_out))

print("output shape ", out.get_shape().as_list())

losses = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=y))

l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.05, scope=None)
weights = tf.trainable_variables() # all vars of your graph
regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)

regularized_loss = losses + regularization_penalty

optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(regularized_loss)

with open('/home/janmejaya/sentiment_files/data/train.pkl', 'rb') as f:
    data = pickle.load(f)

perm = np.arange(len(data['input']))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        print('Training Batch {0} outof 5'.format(i))
        np.random.shuffle(perm)
        input_data = data['input'][perm]
        target_data = data['target'][perm]
        print(input_data.max(1).max())
        print(input_data.shape)

        for idx, data in enumerate(input_data):
            if target_data[idx] == 1:
                target = np.ones([max_word, 1])
            else:
                target = np.zeros([max_word, 1])
            feed_dict = {x: sess.run(tf.one_hot(data, vocab_len)), y: target}
            _, reg_loss, loss = sess.run([optimizer, regularized_loss, losses], feed_dict=feed_dict)
            print('Reg los ', reg_loss)
            if idx % 10000 == 0:
                print('Saving Weightss')
                w_val = sess.run(w_nn1)
                print('Shape ', w_val.shape)
                np.savetxt('w.txt', np.sum(w_val, axis=1))
