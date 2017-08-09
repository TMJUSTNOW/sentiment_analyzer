from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Masking, LSTM, Flatten, Dropout
from keras.datasets import imdb
import numpy as np
from nltk.tokenize import word_tokenize
import tensorflow as tf
from gensim.models.keyedvectors import KeyedVectors
from keras import backend as K

import pickle
import os
import time




# Parameters
max_word = 15
max_features = 25000
batch_size = 100
state_size = 15
n_classes = 2


## Defining custom metrics
def precision(y_true, y_pred):

    # true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # pred_positive = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # precision_val = true_positive / (pred_positive + K.epsilon())
    val = K.sum(K.cast(K.equal(K.cast(K.argmax(y_pred, axis=1), tf.float32), K.cast(K.argmax(y_true, axis=1), tf.float32)), tf.float32))
    return val


def recall(y_true, y_pred):
    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    total_positive = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_val = true_positive / (total_positive + K.epsilon())

    return recall_val


with open('/home/john/sentiment_files/data/movie_vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)


print('Build model...')
model = Sequential()
model.add(Masking(mask_value=0, input_shape=(max_word, 300)))
# model.add(Embedding(max_features, 1000))
model.add(LSTM(state_size, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
model.add(LSTM(state_size, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', precision])




start_idx = 0

def load_data(test):
    if test:
        with open('/home/john/sentiment_files/data/test.pkl', 'rb') as f:
            data = pickle.load(f)
    else:
        with open('/home/john/sentiment_files/data/train.pkl', 'rb') as f:
            data = pickle.load(f)
        # train_dir = '/home/john/sentiment_files/data/complete_data_15_word/train'
        # input_data = np.array([], np.float64)
        # target = np.array([], np.float64)
        # file_list = [each_file for each_file in os.listdir(train_dir)
        #              if os.path.isfile(os.path.join(train_dir, each_file))]
        # for each_file in file_list:
        #     with open(os.path.join(train_dir, each_file), 'rb') as f:
        #         data = pickle.load(f)
        #     if len(input_data) != 0:
        #         input_data = np.vstack((input_data, data['input']))
        #         target = np.hstack((target, data['target']))
        #     else:
        #         input_data = data['input']
        #         target = data['target']
    return data


def data_batch(data, batch_size):
    len_input = len(data['input'])
    perm = np.arange(len_input)
    while True:
        global start_idx
        if start_idx+batch_size >= len_input:
            start_idx = 0
            np.random.shuffle(perm)
            data['input'] = data['input'][perm]
            data['target'] = data['target'][perm]

        input_data = np.zeros([batch_size, max_word, 300], np.float64)
        for id1, each_data in enumerate(data['input'][start_idx:start_idx + batch_size]):
            for id2, word_id in enumerate(each_data):
                word = vocab_data['index_to_vocab'][word_id]
                try:
                    input_data[id1, id2] = word_vectors[word]
                except Exception as exc:
                    continue  # if word doesn't present leave it as zero and mask it(it also include <pad>)
        sess = tf.Session()
        target = sess.run(tf.one_hot(data['target'][start_idx:start_idx + batch_size], n_classes))
        start_idx += batch_size
        yield (input_data, target)



word_vectors = KeyedVectors.load_word2vec_format('/home/john/geek_stuff/Data_Set/Google_News_corpus/GoogleNews-vectors-negative300.bin', binary=True)



# for i in range(10):
#     data['input'] = data['input'][perm]
#     data['target'] = data['target'][perm]
#     while start_idx < len_input:
#         # input_data = np.zeros([batch_size, max_word, 300], np.float64)
#         # for id1, each_data in enumerate(data['input'][start_idx:start_idx+batch_size]):
#         #     for id2, word_id in enumerate(each_data):
#         #         word = vocab_data['index_to_vocab'][word_id]
#         #         try:
#         #             input_data[id1,id2] = word_vectors[word]
#         #         except Exception as exc:
#         #             continue            # if word doesn't present leave it as zero and mask it(it also include <pad>)
#         # sess = tf.Session()
#         # target = sess.run(tf.one_hot(data['target'][start_idx:start_idx+batch_size], n_classes))
#         print('Training on Batches')
#         data_generator = data_batch(batch_size)
#         los = model.train_on_batch(data_generator[0], data_generator[1])
#         print("Loss {0}".format(los))
data = load_data(test=False)
len_input = len(data['input'])
data_generator = data_batch(data, batch_size)
model.fit_generator(data_generator, steps_per_epoch=len_input//batch_size, nb_epoch=7)

data = load_data(test=True)
data_generator = data_batch(data, batch_size)
score, acc, prec = model.evaluate_generator(data_generator, steps=len_input//batch_size)
print('ERROR: {0}, Accuracy: {1}, Precision {2}'.format(score, acc, prec))

# serialize model to JSON
model_json = model.to_json()
with open('/home/john/sentiment_files/model/movie_pre_trained.json', "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/home/john/sentiment_files/model/movie_pre_trained.h5")
print("Saved model to disk")