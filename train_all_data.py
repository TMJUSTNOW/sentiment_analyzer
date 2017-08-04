
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Masking, LSTM, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.datasets import imdb
import numpy as np
from nltk.tokenize import word_tokenize
import tensorflow as tf

import pickle
import os
import time
import random

## Model Parameters
max_word = 15
max_features = 25000
batch_size = 100
state_size = 15
n_classes = 2

train_step = 10


## Model
print('Build model...')
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(max_word, )))
model.add(Embedding(max_features, 1500))
model.add(LSTM(state_size, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))


# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_dir = '/home/john/sentiment_files/data/complete_data_15_word/train'
input_data = np.array([], np.float64)
target = np.array([], np.float64)

file_list = [each_file for each_file in os.listdir(train_dir)
                        if os.path.isfile(os.path.join(train_dir, each_file))]
for each_file in file_list:
    with open(os.path.join(train_dir, each_file), 'rb') as f:
        data = pickle.load(f)
    print('Shape of data ', data['input'].shape)
    print('Shape of target ', data['target'].shape)
    if len(input_data) != 0:
        input_data = np.vstack((input_data, data['input']))
        target = np.hstack((target, data['target']))
    else:
        input_data = data['input']
        target = data['target']

    print('Shape of Input data ', input_data.shape)
    print('Shape id final target ', target.shape)
sess = tf.Session()
target = sess.run(tf.one_hot(target, n_classes))
sess.close()
model.fit(input_data, target, batch_size=batch_size, epochs=3, validation_split=0.3, verbose=1)


print('Testing file...')
with open('/home/john/sentiment_files/data/complete_data_15_word/test/test.pkl', 'rb') as f:
    data = pickle.load(f)
perm = np.arange(len(data['input']))
np.random.shuffle(perm)
sess = tf.Session()
target = sess.run(tf.one_hot(data['target'][perm], n_classes))
sess.close()
score, acc = model.evaluate(data['input'][perm], target, batch_size=batch_size)

print('Final Score {0} and accuracy {1}'.format(score, acc))

## Save model
model_json = model.to_json()
with open('/home/john/sentiment_files/model/complete_sentiment_15_word_new.json', "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/home/john/sentiment_files/model/complete_sentiment_15_word_new.h5")
print("Saved model to disk")
