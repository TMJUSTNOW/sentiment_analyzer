import os
import json
import numpy as np
import html.parser
import re
import pickle
import time

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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

idx_to_vocab = {}
vocab_to_idx = {}
vocab_frequency = {}
len_vocab = 0
index_to_start = 1      # 0th index is used for padding

for each_file in file_list:
    try:
        with open(os.path.join(train_pos_sample_dir, each_file), 'r') as f:
            data = f.read()
    except:
        try:
            with open(os.path.join(train_neg_sample_dir, each_file), 'r') as f:
                data = f.read()
        except:
            try:
                with open(os.path.join(test_neg_sample_dir, each_file), 'r') as f:
                    data = f.read()
            except:
                with open(os.path.join(test_pos_sample_dir, each_file), 'r') as f:
                    data = f.read()

    # PreProcess the data
    # Escape HTML char ir present
    html_parser = html.parser.HTMLParser()
    html_cleaned_data = html_parser.unescape(data)


    # Keep important punctuation
    html_cleaned_data = re.sub('[^A-Za-z ]+', '', html_cleaned_data)

    # TODO: Perform Word Lemmatization on text

    # Break data into words and remove stop Words(or unnecessary words which doesn't have much influence on meaning)
    # TODO: There are few words like 'br', 'o', 'or' after listed to remove still appears in sentence
    stop_words = ['after', 'others', 'clud', 'would', 'id', 'youll', 'men', 'be', 'yourselves', 'im', 'with', 'only', 'is', 'hadn', 'll', 'into', 'being', 'a', 'me', 'has', 'been', 'why', 'where', 'up', 'can', 'than', 'in', 'from', 'who', 'the', 'should', 'your', 'themselves', 'while', 'we', 'against', 'further', 'these', 'those', 'were', 'be', 'out', 'him', 'our', 'which', 'just', 'doing', 'this', 'had', 'theirs', 'i', 'under', 'm', 'herself', 'by', 'at', 'until', 'here', 'were', 'if', 'myself', 'ma', 'ours', 'as', 'all', 'each', 'how', 'when', 'other', 'itself', 'an', 'my', 'did', 'them', 'ourselves', 'during', 'whom', 'am', 'o', 'same', 'are', 'have', 'above', 'what', 'both', 'do', 'off', 'before', 'or', 'having', 'now', 'too', 'on', 'and', 'through', 'she', 'his', 'do', 'there', 'won', 'they', 'to', 'are', 't', 'few', 'was', 'it', 'did', 'himself', 'her', 'some', 'such', 'have', 'yours', 'more', 'for', 're', 'will', 'ain', 's', 'you', 'their', 'about', 'between', 'that', 'once', 'does', 'shall', 'no', 've', 'he', 'of', 'y', 'then', 'own', 'again', 'd', 'any', 'not', 'does', 'was', 'its', 'below', 'hers', 'most', 'but', 'so', 'down', 'yourself']
    movie_stop_words = ['actors', 'story', 'villain', 'chemistry', 'bill', 'lee', 'audiences', 'producers', 'filmed', 'review', 'song', 'musical', 'songs', 'thriller', 'theater', 'br', 'dialog', 'james', 'one', 'first', 'four', 'five', 'ten', 'gore', 'ru', 'filmbr', 'viewers', 'dr', 'era','stefan', 'jrs', 'mar', 'palestinian', 'gandolfini', 'elisha', 'taboos', 'jessie', 'hindu', 'gandolfini', 'gandolfini', 'water', 'de', 'mov', 'moviebr', 'robert', 'v', 'liv', 'david', 'w', 'la', 'sitt', 'volv', 'tak', 'michael', 'ope', 'mr', 'hour', 'movie', 'music', 'film', 'two', 'th', 'youre', 'john', 'bor', 'ett', 'hollywood', 'drama', 'theyre', 'yeah', 'com', 'itbr', 'genre']
    tokenized_word = word_tokenize(html_cleaned_data)
    filtered_sentences = [w.lower() for w in tokenized_word if w.lower() not in stop_words+movie_stop_words]


    # Convert data into standard lexicons
    # Remove 'ing'
    # TODO: Look of any other library doing it or try to add few other to list
    removing_string = ['ing']
    for val in removing_string:
        filtered_sentences = [w.strip(val) if w.endswith(val) else w for w in filtered_sentences]

    # TODO: Add low frequency words from classes
    # create word to index and index to word dictionary
    vocab = set(filtered_sentences)
    for indx, val in enumerate(vocab):
        if val not in vocab_to_idx:
            vocab_to_idx[val] = len_vocab + index_to_start
            idx_to_vocab[len_vocab + index_to_start] = val
            len_vocab += 1

        # Count frequency of the Word
        if val in vocab_frequency:
            vocab_frequency[val] += 1
        else:
            vocab_frequency[val] = 1
    # TODO: Check most frequent occurring words in

# Add Extra Padding words
vocab_to_idx['<pad>'] = 0
idx_to_vocab[0] = '<pad>'
# Save data to pickle object
print("Length of Vocab", len_vocab)

frequent_words_tuple = sorted(vocab_frequency.items(), key=lambda x: x[1], reverse=True)[:1000]
frequent_words = [val for val, freq in frequent_words_tuple]
print(frequent_words)
print('Saving Dictionary')
dictionary = {'vocab_to_index': vocab_to_idx, 'index_to_vocab': idx_to_vocab, 'vocab_frequency': vocab_frequency}
with open('data/movie_vocab.pkl', 'wb') as f:
    pickle.dump(dictionary, f)



# Convert data in to index and store it
print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
with open('data/movie_vocab.pkl', 'rb') as f:
    data = pickle.load(f)
print(len(data['vocab_to_index']))
vocab_to_index = data['vocab_to_index']
index_to_vocab = data['index_to_vocab']
vocab_frequency = data['vocab_frequency']
vocab_len = len(data['vocab_to_index'])
print('Vocab len: ', vocab_len)

max_word = 50
max_features = 20000
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

    # Keep maximum frequent occurring words
    frequent_words_tuple = sorted(vocab_frequency.items(), key=lambda x: x[1], reverse=True)[:max_features]
    frequent_words = [val for val, freq in frequent_words_tuple]

    for index, each_file in enumerate(pos_file_to_read+neg_file_to_read):
        if index % 5000 == 0:
            print(index)
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
            truncated_review += [0] * (max_word - word_len)
        input_data[index] = truncated_review
    return (input_data, target_data)

data_x, data_y = next_batch(25000)
train_dict = {'input': data_x, 'target': data_y}

with open('data/train.pkl', 'wb') as f:
    pickle.dump(train_dict, f)

data_x, data_y = next_batch(25000, test=True)
test_dict = {'input': data_x, 'target': data_y}

with open('data/test.pkl', 'wb') as f:
    pickle.dump(test_dict, f)