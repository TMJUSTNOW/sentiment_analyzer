from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import numpy as np
from nltk.tokenize import word_tokenize
import tensorflow as tf

import pickle
import os


print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
with open('data/movie_vocab.pkl', 'rb') as f:
    data = pickle.load(f)
print(len(data['vocab_to_index']))
vocab_to_index = data['vocab_to_index']
index_to_vocab = data['index_to_vocab']
vocab_len = len(data['vocab_to_index'])
vocab_to_index['<pad>'] = vocab_len
index_to_vocab[vocab_len] = '<pad>'
del data

# Convert data into word vector
max_word = 200
train_pos_sample_dir = '/home/john/geek_stuff/Data Set/IMDB_sentiment_analysis_data/aclImdb/train/pos'
train_neg_sample_dir = '/home/john/geek_stuff/Data Set/IMDB_sentiment_analysis_data/aclImdb/train/neg'
test_neg_sample_dir = '/home/john/geek_stuff/Data Set/IMDB_sentiment_analysis_data/aclImdb/test/neg'
test_pos_sample_dir = '/home/john/geek_stuff/Data Set/IMDB_sentiment_analysis_data/aclImdb/test/pos'

file_list_pos = [each_file for each_file in os.listdir(train_pos_sample_dir) if os.path.isfile(os.path.join(train_pos_sample_dir, each_file))]
file_list_neg = [each_file for each_file in os.listdir(train_neg_sample_dir) if os.path.isfile(os.path.join(train_neg_sample_dir, each_file))]
file_list = file_list_neg + file_list_pos

file_pointer = 0
def next_batch(batch_size, test=False):
    # This function reads equal amount of positive and negative review depending on batch size.
    # returns input_data of size [batch_size, max_chars]
    #     and target_data of size [batch_size, 1]

    global file_pointer, file_list_pos, file_list_neg
    if batch_size % 2 != 0:
        print("[ERROR]Please provide batch_size divisible by 2")
    if test:
        file_list_pos = [each_file for each_file in os.listdir(test_pos_sample_dir) if
                         os.path.isfile(os.path.join(test_pos_sample_dir, each_file))]
        file_list_neg = [each_file for each_file in os.listdir(test_neg_sample_dir) if
                         os.path.isfile(os.path.join(test_neg_sample_dir, each_file))]
    else:
        file_list_pos = [each_file for each_file in os.listdir(train_pos_sample_dir) if
                         os.path.isfile(os.path.join(train_pos_sample_dir, each_file))]
        file_list_neg = [each_file for each_file in os.listdir(train_neg_sample_dir) if
                         os.path.isfile(os.path.join(train_neg_sample_dir, each_file))]

    pos_file_to_read = file_list_pos[file_pointer:(file_pointer + (batch_size // 2))]
    neg_file_to_read = file_list_neg[file_pointer:(file_pointer + (batch_size // 2))]

    if (file_pointer+(batch_size//2)) > len(file_list_pos):
        file_pointer = 0
    else:
        file_pointer += batch_size//2

    data_len_list, data_list = [], []
    target_data = np.zeros([batch_size], dtype=np.int32)
    input_data = np.zeros([batch_size, max_word], dtype=np.int32)
    for index, each_file in enumerate(pos_file_to_read+neg_file_to_read):
        if test:
            pos_dir = test_pos_sample_dir
            neg_dir = test_neg_sample_dir
        else:
            pos_dir = train_pos_sample_dir
            neg_dir = train_neg_sample_dir

        try:
            with open(os.path.join(pos_dir, each_file), 'r') as f:
                data = f.read()
                target_data[index] = 1
        except:
            with open(os.path.join(neg_dir, each_file), 'r') as f:
                data = f.read()
        # Tokenize the data
        tokenized_word = word_tokenize(data)
        truncated_review = []
        for each_word in tokenized_word:
            # Go through each word and discard words which are not present in dictionary
            if each_word in vocab_to_index:
                truncated_review.append(vocab_to_index[each_word])
                if len(truncated_review) >= max_word:
                    break
        # Pad appropriately if less words are present
        word_len = len(truncated_review)
        if word_len < max_word:
            truncated_review += [vocab_len] * (max_word - word_len)
        input_data[index] = truncated_review

    return dict(input_data=input_data, target_data=target_data)

batch_size = 32
state_size = 128

print('Build model...')
model = Sequential()
model.add(Embedding(vocab_len+1, 128))
model.add(LSTM(state_size, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
data = next_batch(25000)
perm = np.arange(25000)
np.random.shuffle(perm)
print('Target\n', data['target_data'][:10])

print("Data Loaded")
model.fit(data['input_data'][perm], data['target_data'][perm], batch_size=batch_size, epochs=2, validation_split=0.3)
data = next_batch(25000, test=True)
score, acc = model.evaluate(data['input_data'][perm], data['target_data'][perm], batch_size=batch_size)


print('Score {0} acc {1}'.format(score, acc))
prediction = model.predict(data['input_data'][perm], batch_size=batch_size)
# print(prediction)
sess = tf.Session()
prediction = sess.run(tf.nn.softmax(prediction))
# print(np.argmax(prediction))
# print(data['target_data'])
acc = np.sum(np.equal(np.argmax(prediction), data['target_data'][perm])) / len(data['target_data'])
print(acc)