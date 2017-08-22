import os
import json
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Flatten


# Read data set(IMDB Review data)
train_pos_sample_dir = '/home/janmejaya/sentiment_analyzer/aclImdb/train/pos'
train_neg_sample_dir = '/home/janmejaya/sentiment_analyzer/aclImdb/train/neg'
test_neg_sample_dir = '/home/janmejaya/sentiment_analyzer/aclImdb/test/neg'
test_pos_sample_dir = '/home/janmejaya/sentiment_analyzer/aclImdb/test/pos'


train_pos = [each_file for each_file in os.listdir(train_pos_sample_dir) if os.path.isfile(os.path.join(train_pos_sample_dir, each_file))]
train_neg = [each_file for each_file in os.listdir(train_neg_sample_dir) if os.path.isfile(os.path.join(train_neg_sample_dir, each_file))]
test_pos = [each_file for each_file in os.listdir(test_pos_sample_dir) if os.path.isfile(os.path.join(test_pos_sample_dir, each_file))]
test_neg = [each_file for each_file in os.listdir(test_neg_sample_dir) if os.path.isfile(os.path.join(test_neg_sample_dir, each_file))]

file_list = train_pos + train_neg + test_pos + test_neg
data = ''
for num_of_file, each_file in enumerate(file_list):
    try:
        with open(os.path.join(train_pos_sample_dir, each_file), 'r') as f:
            data += f.read()
    except:
        try:
            with open(os.path.join(train_neg_sample_dir, each_file), 'r') as f:
                data += f.read()
        except:
            try:
                with open(os.path.join(test_neg_sample_dir, each_file), 'r') as f:
                    data += f.read()
            except:
                with open(os.path.join(test_pos_sample_dir, each_file), 'r') as f:
                    data += f.read()

vocab = set(data)
len_vocab = len(vocab)
del file_list
del data
print('Len of vocab', len_vocab)
idx_to_vocab = dict(enumerate(vocab))
vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))
# Adding Extra Padding char to dictionary
idx_to_vocab[len_vocab] = '<pad>'
vocab_to_idx['<pad>'] = len_vocab
pad_index = len_vocab

# Data Loader
test_pointer = 0
train_pointer = 0
max_chars = 50     # Maximum number of character to keep in each Review
def next_batch(batch_size, test=False):
    # This function reads equal amount of positive and negative review depending on batch size.
    # returns input_data of size [batch_size, max_chars] and target_data of size [batch_size, 1]

    global test_pointer, train_neg, train_pos, len_vocab, test_pos, test_neg, train_pointer
    if batch_size % 2 != 0:
        print("[ERROR]Please provide batch_size divisible by 2")
    if test:
        pos_file_to_read = test_pos[test_pointer:(test_pointer + (batch_size // 2))]
        neg_file_to_read = test_neg[test_pointer:(test_pointer + (batch_size // 2))]
        if (test_pointer + (batch_size // 2)) > len(test_pos+test_neg):
            test_pointer = 0
        else:
            test_pointer += batch_size // 2
    else:
        pos_file_to_read = train_pos[train_pointer:(train_pointer + (batch_size // 2))]
        neg_file_to_read = train_neg[train_pointer:(train_pointer + (batch_size // 2))]
        if (train_pointer + (batch_size // 2)) > len(train_pos+train_neg):
            train_pointer = 0
        else:
            train_pointer += batch_size // 2


    input_data = np.zeros([batch_size, max_chars], dtype=np.int32)
    target_data = np.zeros([batch_size], dtype=np.int32)
    char_len = np.zeros([batch_size], dtype=np.int32)

    for position, each_file in enumerate(pos_file_to_read+neg_file_to_read):
        if test:
            pos_dir = test_pos_sample_dir
            neg_dir = test_neg_sample_dir
        else:
            pos_dir = train_pos_sample_dir
            neg_dir = train_neg_sample_dir
        try:
            with open(os.path.join(pos_dir, each_file), 'r') as f:
                data = f.read()
                if len(data) < max_chars:
                    number_of_padding = (max_chars - len(data))
                    char_len[position] = len(data)
                else:
                    number_of_padding = 0
                    char_len[position] = max_chars

                truncated_review = []
                for char in data[:max_chars]:
                    if char not in vocab_to_idx:
                        len_vocab += 1
                        vocab_to_idx[char] = len_vocab
                    truncated_review.append(vocab_to_idx[char])

                if number_of_padding:
                    input_data[position] = truncated_review + [pad_index] * number_of_padding
                else:
                    input_data[position] = truncated_review
                target_data[position] = 1
        except:
            with open(os.path.join(neg_dir, each_file), 'r') as f:
                data = f.read()
                if len(data) < max_chars:
                    number_of_padding = (max_chars - len(data))
                    char_len[position] = len(data)
                else:
                    number_of_padding = 0
                    char_len[position] = max_chars

                truncated_review = []
                for char in data[:max_chars]:
                    if char not in vocab_to_idx:
                        len_vocab += 1
                        vocab_to_idx[char] = len_vocab
                    truncated_review.append(vocab_to_idx[char])

                if number_of_padding:
                    input_data[position] = truncated_review + [pad_index] * number_of_padding
                else:
                    input_data[position] = truncated_review
                target_data[position] = 0

    return dict(input_data=input_data, target_data=target_data, sequence_len=char_len)

# Model Parameters
batch_size = 25000
num_classes = 2
state_size = 400
learning_rate = 1e-2
disp_step = 50
saving_step = 10
fc_neurons = 1000
num_layers = 2

# Build Model
model = Sequential()
model.add(LSTM(state_size, return_sequences=True, input_shape=(max_chars, len_vocab)))
model.add(LSTM(state_size, return_sequences=True))
model.add(LSTM(state_size, return_sequences=True))
# model.add(Dense(fc_neurons, activation='relu'))
model.add(Flatten())
model.add(Dense(num_classes))

model.compile(optimizer='Adagrad', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

for epoch in range(1200):        # Number of iteration over complete data
    # for iterate_over_data in range(len(train_pos) // batch_size):  # Iterate through data
    # get next batch of data
    data_dict = next_batch(batch_size=batch_size)
    sess = tf.Session()
    input_data = sess.run(tf.one_hot(data_dict['input_data'], len_vocab))
    # target_data = sess.run(tf.one_hot(data_dict['target_data'], num_classes))
    sess.close()
    model.fit(x=input_data, y=data_dict['target_data'], batch_size=100, epochs=100,  validation_split=0.1)

    if (epoch % disp_step) == 0:
        for iterate_over_data in range(len(train_pos) // batch_size):  # Iterate through data
            data_dict = next_batch(batch_size=batch_size, test=True)
            sess = tf.Session()
            input_data = sess.run(tf.one_hot(data_dict['input_data'], len_vocab))
            target_data = sess.run(tf.one_hot(data_dict['target_data'], num_classes))
            sess.close()
            result = model.evaluate(x=input_data, y=data_dict['target_data'], batch_size=batch_size)
            print("Result: ", result)

