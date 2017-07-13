"""
This model tries to analyze sentiment (at character level) using LSTM
Read this Example of RNN in TF
https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
"""
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import mode

## Data preprocessing

# Read data set(IMDB Review data)
train_pos_sample_dir = '/home/john/geek_stuff/Data Set/IMDB_sentiment_analysis_data/aclImdb/train/pos'
train_neg_sample_dir = '/home/john/geek_stuff/Data Set/IMDB_sentiment_analysis_data/aclImdb/train/neg'
test_neg_sample_dir = '/home/john/geek_stuff/Data Set/IMDB_sentiment_analysis_data/aclImdb/test/neg'
test_pos_sample_dir = '/home/john/geek_stuff/Data Set/IMDB_sentiment_analysis_data/aclImdb/test/pos'

# create a set of all character
file_list_pos = [each_file for each_file in os.listdir(train_pos_sample_dir) if os.path.isfile(os.path.join(train_pos_sample_dir, each_file))]
file_list_neg = [each_file for each_file in os.listdir(train_neg_sample_dir) if os.path.isfile(os.path.join(train_neg_sample_dir, each_file))]
file_list = file_list_neg + file_list_pos
data = ''
for num_of_file, each_file in enumerate(file_list):
    try:
        with open(os.path.join(train_pos_sample_dir, each_file), 'r') as f:
            data += f.read()
    except:
        with open(os.path.join(train_neg_sample_dir, each_file), 'r') as f:
            data += f.read()
    if num_of_file > 3000:
        break

# Generating vocabulary of unique Character.
vocab = set(data)
len_vocab = len(vocab)
del file_list
del data
print('Len of vocab', len_vocab)
idx_to_vocab = dict(enumerate(vocab))
vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))

file_pointer = 0
def next_batch(batch_size, test=False):
    # This function reads equal amount of positive and negative review depending on batch size.
    # returns input_data of size [batch_size, max_chars]
    #     and target_data of size [batch_size, 1]

    global file_pointer, file_list_pos, file_list_neg, len_vocab
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
        data_len_list.append(len(data))
        data_list.append(data)

    # Finding Most common length
    max_chars = mode(data_len_list)[0][0]

    input_data = np.zeros([batch_size, max_chars], dtype=np.int32)

    for i in range(len(data_list)):
        data = data_list.pop(0)
        if len(data) < max_chars:
            number_of_padding = (max_chars-len(data))
        else:
            number_of_padding = 0

        truncated_review = []
        for char in data[:max_chars]:
            if char not in vocab_to_idx:
                print('Vocan len in ', len_vocab)
                print('max value ', max(vocab_to_idx.values()))
                len_vocab += 1
                vocab_to_idx[char] = len_vocab
            truncated_review.append(vocab_to_idx[char])

        if number_of_padding:
            input_data[i] = truncated_review + ['<pad>']*number_of_padding
        else:
            input_data[i] = truncated_review

    return dict(input_data=input_data, target_data=target_data)


batch_size = 100
num_classes = 2
state_size = 50
learning_rate = 1e-2
disp_step = 50
saving_step = 10
fc_neurons = 100
restore_model = False
# whf = tf.Variable(tf.random_normal([max_chars*state_size, fc_neurons]), name='whf')          # weights from hidden layer to fc layer
# bhf = tf.Variable(tf.random_normal([fc_neurons]), name='bhf')
# wfo = tf.Variable(tf.random_normal([fc_neurons, num_classes], name='wfo'))      # Weight from fully connected to output layer
# bfo = tf.Variable(tf.random_normal([num_classes], name='bfo'))
init_hidden_state_c = tf.Variable(tf.random_normal([batch_size, state_size]), name='init_hidden_state_c')
init_hidden_state_h = tf.Variable(tf.random_normal([batch_size, state_size]), name='init_hidden_state_h')


x = tf.placeholder(tf.int32, [batch_size, None], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size], name='labels_placeholder')

# input to dynamic RNN should be of shape [batch_size, len_of_seq, input_size]
rnn_inputs = tf.one_hot(x, len_vocab)
print("onehot shape ", rnn_inputs.get_shape().as_list())
cell = tf.contrib.rnn.BasicLSTMCell(state_size)
if restore_model:
    print("\nRestoring Model\n")
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("tf_models/char_level_sentiment_analyzer.meta")
        saver.restore(sess, 'tf_models/char_level_sentiment_analyzer')
        whf = tf.Variable(tf.get_default_graph().get_tensor_by_name('whf:0'), name='whf1')
        bhf = tf.Variable(tf.get_default_graph().get_tensor_by_name('bhf:0'), name='bhf1')
        wfo = tf.Variable(tf.get_default_graph().get_tensor_by_name('wfo:0'), name='wfo1')
        bfo = tf.Variable(tf.get_default_graph().get_tensor_by_name('bfo:0'), name='bfo1')
        # hidden_state_c1 = tf.Variable(tf.get_default_graph().get_tensor_by_name('init_hidden_state_c:0'), name='init_hidden_state_c1', trainable=False)
        # hidden_state_h1 = tf.Variable(tf.get_default_graph().get_tensor_by_name('init_hidden_state_h:0'),
        #                              name='init_hidden_state_h1', trainable=False)
        init_state = tuple([tf.Variable(tf.get_default_graph().get_tensor_by_name('init_hidden_state_c:0'),
                                        name='init_hidden_state_c1', trainable=False),
                            tf.Variable(tf.get_default_graph().get_tensor_by_name('init_hidden_state_h:0'),
                                        name='init_hidden_state_h1', trainable=False)])
else:
    whf = tf.Variable(tf.random_normal([max_chars * state_size, fc_neurons]),
                      name='whf')  # weights from hidden layer to fc layer
    bhf = tf.Variable(tf.random_normal([fc_neurons]), name='bhf')
    wfo = tf.Variable(
        tf.random_normal([fc_neurons, num_classes], name='wfo'))  # Weight from fully connected to output layer
    bfo = tf.Variable(tf.random_normal([num_classes], name='bfo'))
    init_state = cell.zero_state(batch_size, tf.float32)
    print("Init state type, ", type(init_state))
    print("Init state [0] type, ", type(init_state[0]))
    print('Init state [0]', init_state[0])

# init_state = tf.Variable(cell.zero_state(batch_size, tf.float32), name='init_state')
rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)
# rnn_outputs is a list of "number of inputs(or max_chars)" of rnn_output
# shape of rnn_output is [batch_size, state_size]
# logits has shape of [batch_size, num_classes]
# Changing Shape of rnn_outputs to [batch_size, max_chars, state_size]
rnn_output = tf.transpose(rnn_outputs, [1, 0, 2])
print("RNN_outputs Shape ", len(rnn_outputs))
dropped_rnn_output = tf.nn.dropout(rnn_output, keep_prob=0.75)

# Flattening output before connecting it to FC layer
flatten_output = tf.reshape(dropped_rnn_output, [-1, whf.get_shape().as_list()[0]])
# Connecting a Fully connected layer on top
fc1 = tf.nn.relu(tf.add(tf.matmul(flatten_output, whf), bhf))
dropped_fc1 = tf.nn.dropout(fc1, keep_prob=0.75)
# O/P layer
out = tf.add(tf.matmul(dropped_fc1, wfo), bfo)


losses =tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=y)
total_loss = tf.reduce_mean(losses)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(indices=y, depth=num_classes)))
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

# Accuracy
correct_pred = tf.equal(tf.cast(tf.argmax(out, 1), tf.int32), y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float64))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    training_state = []
    for epoch in range(1200):        # Number of iteration over complete data
        total_loss_data_val = 0
        for iterate_over_data in range(len(file_list_pos)//batch_size):   # Iterate through data
            # get next batch of data
            data_dict = next_batch(batch_size=batch_size)
            feed_dict = {x: data_dict['input_data'], y: data_dict['target_data']}
            if len(training_state) != 0:
                feed_dict[init_state] = training_state
            _, total_loss_data, training_state, disp_out_put = sess.run([optimizer, total_loss, final_state, out], feed_dict=feed_dict)
            tf.assign(ref=init_hidden_state_c, value=training_state[0])
            tf.assign(ref=init_hidden_state_h, value=training_state[1])
            total_loss_data_val += total_loss_data
        print("Total Loss after {0}th epoch over data: {1}".format(epoch, (
                                                                    total_loss_data_val / (len(file_list_pos)//batch_size))))

        if epoch % saving_step == 0:
            saver = tf.train.Saver({'init_hidden_state_c': init_hidden_state_c, 'init_hidden_state_h': init_hidden_state_h,
                                    'whf': whf, 'bhf': bhf, 'wfo': wfo, 'bfo': bfo})
            saver.save(sess, 'tf_models/char_level_sentiment_analyzer')
            print('Model saved, step: {0}'.format(epoch))
        if epoch % disp_step == 0:
            # Finding Testing Error
            print('Testing For data')
            acc = 0
            for iterate_over_data in range(len(file_list_pos) // batch_size):
                data_dict = next_batch(batch_size, test=True)
                feed_dict2 = {x: data_dict['input_data'], y: data_dict['target_data']}
                acc += sess.run([accuracy], feed_dict=feed_dict2)[0]
            print('Accuracy in percentage: {0}'.format((acc / (len(file_list_pos) // batch_size)) * 100))

            # sample the result
            # size of logits [num_char, batch_size, n_class]
            data_dict = next_batch(batch_size, test=True)
            feed_dict2 = {x: data_dict['input_data']}
            if len(training_state) != 0:
                feed_dict2[init_state] = training_state
            prediction = sess.run([out], feed_dict=feed_dict2)
            batch_prediction = sess.run(tf.nn.softmax(prediction[0]))
            review = ''
            for ind in data_dict['input_data'][0]:
                review += idx_to_vocab[ind]
            print(review)
            print('\nPrediction\t', batch_prediction[0])
            print('\nActual Output\t', data_dict['target_data'][0])
            review = ''
            for ind in data_dict['input_data'][batch_size-1]:
                review += idx_to_vocab[ind]
            print(review)
            print('\nPrediction\t', batch_prediction[batch_size-1])
            print('\nActual Output\t', data_dict['target_data'][batch_size-1])