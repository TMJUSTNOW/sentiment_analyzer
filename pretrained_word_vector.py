from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Masking, LSTM, Flatten, Dropout, TimeDistributed, Lambda
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


# Model Parameters
max_word = 15
batch_size = 32
state_size = 60
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

def sum_pooling(x):
    return K.sum(x, axis=1)


def get_len_of_data(data_dir):
    len_input = 0
    file_list = [each_file for each_file in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, each_file))]
    for each_file in file_list:
        with open(os.path.join(data_dir, each_file), 'rb') as f:
            data = pickle.load(f)
            len_input += len(data['input'])

    return len_input

def extract_data_batch_from_dir(data_dir, batch_size, vocab_dir, word2vec_dir):


    # Load google's Pre-trained word to vector
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_dir, binary=True)
    # Load Vocab
    with open(vocab_dir, 'rb') as f:
        vocab_data = pickle.load(f)

    # len_input = len(data['input'])
    # perm = np.arange(len_input)
    file_list = [each_file for each_file in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, each_file))]
    global start_idx, each_file_data_len
    start_idx = 0
    each_file_data_len = 0
    file_list_len = len(file_list)
    n_file_to_combine = file_list_len
    while True:
        if start_idx+batch_size >= each_file_data_len:
            # Read from a New file
            start_idx = 0

            if file_list_len >= n_file_to_combine:
                file_to_combine = [file_list[i] for i in np.random.choice(file_list_len, n_file_to_combine, replace=False)]
            else:
                file_to_combine = file_list

            print('processing file: {0}'.format(file_to_combine))
            for idx, file in enumerate(file_to_combine):
                with open(os.path.join(data_dir, file), 'rb') as f:
                    data = pickle.load(f)
                if idx == 0:
                    data_input = data['input']
                    data_target = data['target']
                else:
                    data_input = np.concatenate((data_input, data['input']), axis=0)
                    data_target = np.concatenate((data_target, data['target']), axis=0)
            # when 'data_input' uninitialized it should throw error
            each_file_data_len = len(data_input)
            perm = np.arange(each_file_data_len)
            np.random.shuffle(perm)
            data['input'] = data_input[perm]
            data['target'] = data_target[perm]

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

def lstm_model():
    print('Building model...')
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(max_word, 300)))
    model.add(TimeDistributed(Dense(150, activation='relu')))
    model.add(Dropout(0.4))
    model.add(LSTM(state_size, dropout=0.4, recurrent_dropout=0.5, activation='relu', return_sequences=False))
    # model.add(Lambda(sum_pooling))
    # model.add(Flatten())
    # model.add(Dropout(0.4))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

    return model

def train_model(train_dir, vocab_dir, word2vec_dir, test_dir=None, check_point_dir=None, saving_dir=None):

    ## Get Model
    model = lstm_model()

    ## Train Model
    # Load/Save data from checkpoint
    if check_point_dir:
        if os.path.exists(check_point_dir):
            # Loading Saved Weights
            model.load_weights(check_point_dir)
        else:
            ## Saving Model at every epoch
            checkpoint = ModelCheckpoint(check_point_dir, verbose=1, save_best_only=False, save_weights_only=True, period=1)
            callbacks_list = [checkpoint]

    # getting number of input samples (help in deciding number of steps to iterate over generator)
    len_train = get_len_of_data(train_dir)
    # get iterator to iterate over train data
    data_generator = extract_data_batch_from_dir(train_dir, batch_size, vocab_dir, word2vec_dir)
    model.fit_generator(data_generator, steps_per_epoch=len_train // batch_size, nb_epoch=8, callbacks=None, verbose=1,
                        validation_data=None, validation_steps=None)

    ## Test Model
    if test_dir:
        len_test = get_len_of_data(test_dir)
        data_generator = extract_data_batch_from_dir(test_dir, batch_size, vocab_dir, word2vec_dir)
        score, acc = model.evaluate_generator(data_generator, steps=len_test // batch_size)
        print('ERROR: {0}, Accuracy: {1}'.format(score, acc))

    ## Save model
    if saving_dir:
        # serialize model to JSON
        model_json = model.to_json()
        with open(saving_dir + '.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(saving_dir + ".h5")
        print("Saved model to {0}".format(saving_dir))


if __name__ == '__main__':
    train_data_dir = '/home/john/sentiment_files/data/cornel_univ_movie_data/train'
    test_data_dir = '/home/john/sentiment_files/data/cornel_univ_movie_data/test'
    check_point_dir = 'complete_pretrained_check_point_weights_dense+LSTM.hdf5'
    saving_dir = '/home/john/sentiment_files/model/complete_pre_trained_dense+LSTM'
    vocab_dir = '/home/john/sentiment_files/data/cornel_univ_movie_data/cornel_univ_movie_vocab.pkl'
    word2vec_dir = '/home/john/geek_stuff/Data_Set/Google_News_corpus/GoogleNews-vectors-negative300.bin'
    train_model(train_dir=train_data_dir, test_dir=test_data_dir, vocab_dir=vocab_dir, word2vec_dir=word2vec_dir,
                check_point_dir=check_point_dir, saving_dir=saving_dir)