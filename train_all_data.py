
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Masking, LSTM
from keras.datasets import imdb
import numpy as np
from nltk.tokenize import word_tokenize
import tensorflow as tf

import pickle
import os
import time

## Model Parameters
max_word = 15
max_features = 25000
batch_size = 32
state_size = 20
n_classes = 2

train_step = 10


## Model
print('Build model...')
model = Sequential()
model.add(Masking(mask_value=0, input_shape=(max_word, )))
model.add(Embedding(max_features, 2000))
model.add(LSTM(state_size, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(2, activation='softmax'))


# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_dir = '/home/janmejaya/sentiment_files/data'
test_dir = '/home/janmejaya/sentiment_files/data'
for i in range(10):         # Number of iteration over complete data
    file_list = [each_file for each_file in os.listdir(train_dir)
                            if os.path.isfile(os.path.join(train_dir, each_file))]
    # Shuffling file in each iteration
    perm = np.arange(len(file_list))
    np.random.shuffle(perm)
    print('File list before ', file_list)
    file_list = file_list[perm]
    print('File list after ', file_list)
    for each_file in file_list:
        with open(os.path.join(train_dir, each_file), 'rb') as f:
            data = pickle.load(f)
        perm = np.arange(len(data['input']))
        np.random.shuffle(perm)
        sess = tf.Session()
        target = sess.run(tf.one_hot(data['target'][perm], n_classes))
        sess.close()
        model.fit(data['input'][perm], target, batch_size=batch_size, epochs=2, validation_split=0.1)

print('Testing file...')
with open('/home/janmejaya/sentiment_files/data/test.pkl', 'rb') as f:
    data = pickle.load(f)
perm = np.arange(len(data['input']))
np.random.shuffle(perm)
sess = tf.Session()
target = sess.run(tf.one_hot(data['target'][perm], n_classes))
sess.close()
score, acc = model.evaluate(data['input'][perm], target, batch_size=batch_size)

print('Final Score {0} and accuracy {1}'.format(score, acc))

## Save model
