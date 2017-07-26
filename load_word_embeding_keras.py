import os
import pickle
import time

from keras.models import model_from_json
import numpy as np
from nltk.tokenize import word_tokenize
import tensorflow as tf



with open('/home/john/sentiment_files/data/complete_data/complete_train1.pkl', 'rb') as f:
    data = pickle.load(f)

print('Input data shape ', data['input'].shape)
print('Target data shape ', data['target'].shape)
print('input data \n', data['input'][:3])
print('input data \n', data['target'][:3])
time.sleep(1000)







print('Loading data...')
with open('data/movie_vocab.pkl', 'rb') as f:
    data = pickle.load(f)
print(len(data['vocab_to_index']))
vocab_to_index = data['vocab_to_index']
index_to_vocab = data['index_to_vocab']
vocab_frequency = data['vocab_frequency']
vocab_len = len(data['vocab_to_index'])
print('Vocab len: ', vocab_len)

train_pos_sample_dir = '/home/john/geek_stuff/Data Set/IMDB_sentiment_analysis_data/aclImdb/train/pos'
train_neg_sample_dir = '/home/john/geek_stuff/Data Set/IMDB_sentiment_analysis_data/aclImdb/train/neg'
test_neg_sample_dir = '/home/john/geek_stuff/Data Set/IMDB_sentiment_analysis_data/aclImdb/test/neg'
test_pos_sample_dir = '/home/john/geek_stuff/Data Set/IMDB_sentiment_analysis_data/aclImdb/test/pos'

file_list_pos = [each_file for each_file in os.listdir(test_pos_sample_dir) if os.path.isfile(os.path.join(test_pos_sample_dir, each_file))]
file_list_neg = [each_file for each_file in os.listdir(test_neg_sample_dir) if os.path.isfile(os.path.join(test_neg_sample_dir, each_file))]
file_list = file_list_neg + file_list_pos

file_pointer = 0
max_features = 20000
max_word = 50
def next_batch(batch_size):
    # This function reads equal amount of positive and negative review depending on batch size.
    # returns input_data of size [batch_size, max_chars]
    #     and target_data of size [batch_size, 1]

    global file_pointer, file_list_pos, file_list_neg
    if batch_size % 2 != 0:
        print("[ERROR]Please provide batch_size divisible by 2")

    pos_file_to_read = file_list_pos[file_pointer:(file_pointer + (batch_size // 2))]
    neg_file_to_read = file_list_neg[file_pointer:(file_pointer + (batch_size // 2))]

    if (file_pointer+(batch_size//2)) > len(file_list_pos):
        file_pointer = 0
    else:
        file_pointer += batch_size//2

    data_len_list, data_list = [], []
    target_data = np.zeros([batch_size], dtype=np.int32)
    input_data = np.zeros([batch_size, max_word], dtype=np.int32)
    original_data = []

    # Keep maximum frequent occurring words
    frequent_words_tuple = sorted(vocab_frequency.items(), key=lambda x: x[1], reverse=True)[:max_features]
    frequent_words = [val for val, freq in frequent_words_tuple]
    print('Length of frequent words: ', frequent_words)

    for index, each_file in enumerate(pos_file_to_read+neg_file_to_read):

        try:
            with open(os.path.join(test_pos_sample_dir, each_file), 'r') as f:
                data = f.read()
                target_data[index] = 1
        except:
            with open(os.path.join(test_neg_sample_dir, each_file), 'r') as f:
                data = f.read()

        original_data.append(data)
        # Tokenize the data
        tokenized_word = word_tokenize(data)
        truncated_review = []
        for each_word in tokenized_word:
            # Go through each word and discard words which are not present in dictionary
            each_word = each_word.lower()
            # Take words which are frequent
            if each_word in frequent_words:
                try:
                    truncated_review.append(vocab_to_index[each_word])
                    if len(truncated_review) >= max_word:
                        break
                except Exception as exc:
                    print('[Exception] if Key word not present in vocab dict but present in frequent words its a bug: {0}'.format(exc))
        # Pad appropriately if less words are present
        word_len = len(truncated_review)
        if word_len < max_word:
            truncated_review += [vocab_len] * (max_word - word_len)
        input_data[index] = truncated_review

    return dict(input_data=input_data, target_data=target_data, original_data=original_data)




# load json and create model
with open('model/imdb_lstm.json', 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model/imdb_lstm.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
data_dict = next_batch(4)
prediction = loaded_model.predict(data_dict['input_data'])
sess = tf.Session()
prediction = np.argmax(sess.run(tf.nn.softmax(prediction)), axis=1)
for i in range(len(data_dict['target_data'])):
    print('Prediction\n', prediction[i])
    print('Original target\n',  data_dict['target_data'][i])
    print('original data\n', data_dict['original_data'][i])
    stri = ''
    for idx in data_dict['input_data'][i]:
        stri += ' ' + index_to_vocab[idx]
    print('Word Used to learn\n', stri)