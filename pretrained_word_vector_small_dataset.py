from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Masking, LSTM, Flatten, Dropout, TimeDistributed
from keras.datasets import imdb
import numpy as np
from nltk.tokenize import word_tokenize
import tensorflow as tf
from gensim.models.keyedvectors import KeyedVectors
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import pickle
import os
import time


# Parameters
max_word = 15
batch_size = 45
state_size = 30
n_classes = 2

word_vectors = KeyedVectors.load_word2vec_format('/home/john/geek_stuff/Data_Set/Google_News_corpus/GoogleNews-vectors-negative300.bin', binary=True)


def get_data(dir):
    files = [each_file for each_file in os.listdir(dir) if os.path.isfile(os.path.join(dir, each_file))]
    print('Files Present ', files)
    input_data = np.array([], dtype=np.float64)
    target_data = np.array([], dtype=np.float64)
    for each_file in files:
        with open(os.path.join(dir, each_file), 'rb') as f:
            data = pickle.load(f)

            for each_sentence in data['input']:
                for each_word in each_sentence:

            if len(input_data) == 0:
                input_data = data['input']
                target_data = data['target']
            else:
                input_data = np.concatenate((input_data, data['input']), axis=0)
                target_data = np.concatenate((target_data, data['target']), axis=0)
    return (input_data,target_data)



print('Build model...')
model = Sequential()
model.add(Masking(mask_value=0, input_shape=(max_word, 300)))
model.add(TimeDistributed(Dense(100, activation='relu')))
model.add(Dropout(0.4))
model.add(LSTM(state_size, dropout=0.4, recurrent_dropout=0.5, activation='relu'))
# model.add(TimeDistributedMerge(mode='sum'))
# model.add(Flatten())
# model.add(Dropout(0.4))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])


train_dir = '/home/john/sentiment_files/data/cornel_univ_movie_data/train'
train_input, train_target = get_data(train_dir)
perm = np.arange(train_input.shape[0])
np.random.shuffle(perm)
model.fit(train_input[perm], train_target[perm], batch_size=batch_size, epochs=5, validation_split=0.3)

test_dir = '/home/john/sentiment_files/data/cornel_univ_movie_data/test'
test_input, test_target = get_data(test_dir)
perm = np.arange(test_input.shape[0])
np.random.shuffle(perm)

score, acc = model.evaluate(test_input[perm], test_target[perm], batch_size=batch_size)

