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
batch_size = 500
state_size = 15
n_classes = 2


## Defining custom metrics
def precision(y_true, y_pred):
    """
    Metrics function for keras. pass function object as list to keras compile
    """
    # true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # pred_positive = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # precision_val = true_positive / (pred_positive + K.epsilon())
    val = K.sum(K.cast(K.equal(K.cast(K.argmax(y_pred, axis=1), tf.float32), K.cast(K.argmax(y_true, axis=1), tf.float32)), tf.float32))
    return val


def recall(y_true, y_pred):
    """
    Metrics function for keras. pass function object as list to keras compile
    """
    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    total_positive = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_val = true_positive / (total_positive + K.epsilon())

    return recall_val


with open('/home/john/sentiment_files/data/complete_data_15_word/complete_vocab_15_word.pkl', 'rb') as f:
    vocab_data = pickle.load(f)


print('Build model...')
model = Sequential()
model.add(Masking(mask_value=0, input_shape=(max_word, 300)))
# model.add(TimeDistributed(Dense(50, activation='relu'), input_shape=(max_word, 300)))
# model.add(Dropout(0.4))
model.add(LSTM(state_size, dropout=0.4, recurrent_dropout=0.5, activation='relu'))
# model.add(Flatten())
# model.add(Dropout(0.4))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Loading Saved Weights
# model.load_weights("complete_pretrained_check_point_weights.hdf5")

## Saving Model at every epoch
filepath="complete_pretrained_check_point_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False, save_weights_only=True, period=1)
callbacks_list = [checkpoint]


def get_len_of_data(data_dir):
    len_input = 0
    file_list = [each_file for each_file in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, each_file))]
    for each_file in file_list:
        with open(os.path.join(data_dir, each_file), 'rb') as f:
            data = pickle.load(f)
            len_input += len(data['input'])

    return len_input

start_idx = 0
each_file_data_len = 0      # initialisation so first time condition will be true
def extract_data_batch_from_dir(data_dir, batch_size):
    # len_input = len(data['input'])
    # perm = np.arange(len_input)
    file_list = [each_file for each_file in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, each_file))]
    global start_idx, each_file_data_len
    start_idx = 0
    each_file_data_len = 0
    while True:
        if start_idx+batch_size >= each_file_data_len:
            # Read from a New file
            start_idx = 0
            data_file = file_list.pop(0)
            file_list += [data_file]          # Storing at end of file so loop will continue indefinitely
            print('processing file: {0}'.format(data_file))
            with open(os.path.join(data_dir, data_file), 'rb') as f:
                data = pickle.load(f)
                each_file_data_len = len(data['input'])
                perm = np.arange(each_file_data_len)
                np.random.shuffle(perm)
                data['input'] = data['input'][perm]
                data['target'] = data['target'][perm]

        input_data = np.zeros([batch_size, max_word, 300], np.float64)
        for id1, each_data in enumerate(data['input'][start_idx:start_idx + batch_size]):
            sentence = ''
            for id2, word_id in enumerate(each_data):
                word = vocab_data['index_to_vocab'][word_id]
                sentence += ' ' + word
                try:
                    input_data[id1, id2] = word_vectors[word]
                except Exception as exc:
                    continue  # if word doesn't present leave it as zero and mask it(it also include <pad>)
        sess = tf.Session()
        target = sess.run(tf.one_hot(data['target'][start_idx:start_idx + batch_size], n_classes))
        print('Sentence: {0}'.format(sentence))
        print('Target: {0}'.format(target[batch_size - 1]))
        start_idx += batch_size
        yield (input_data, target)


word_vectors = KeyedVectors.load_word2vec_format('/home/john/geek_stuff/Data_Set/Google_News_corpus/GoogleNews-vectors-negative300.bin', binary=True)


train_dir = '/home/john/sentiment_files/data/complete_data_15_word/train'
len_input = get_len_of_data(train_dir)
print('Length of input data {0}'.format(len_input))
data_generator = extract_data_batch_from_dir(train_dir, batch_size)
model.fit_generator(data_generator, steps_per_epoch=len_input//batch_size, nb_epoch=3, callbacks=callbacks_list)

test_dir = '/home/john/sentiment_files/data/complete_data_15_word/test'
len_input = get_len_of_data(test_dir)
print('Length of test data {0}'.format(len_input))
data_generator = extract_data_batch_from_dir(test_dir, batch_size)
score, acc = model.evaluate_generator(data_generator, steps=len_input//batch_size)
print('ERROR: {0}, Accuracy: {1}'.format(score, acc))

# serialize model to JSON
model_json = model.to_json()
with open('/home/john/sentiment_files/model/complete_pre_trained.json', "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/home/john/sentiment_files/model/complete_pre_trained.h5")
print("Saved model to disk")


# for data in extract_data_batch_from_dir('/home/john/sentiment_files/data/complete_data_15_word/train', batch_size):
#     continue