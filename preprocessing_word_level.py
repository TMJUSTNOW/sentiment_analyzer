import os
import json
import numpy as np
import html.parser
import re
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Read data set(IMDB Review data)
train_pos_sample_dir = '/home/john/geek_stuff/Data Set/IMDB_sentiment_analysis_data/aclImdb/train/pos'
train_neg_sample_dir = '/home/john/geek_stuff/Data Set/IMDB_sentiment_analysis_data/aclImdb/train/neg'
test_neg_sample_dir = '/home/john/geek_stuff/Data Set/IMDB_sentiment_analysis_data/aclImdb/test/neg'
test_pos_sample_dir = '/home/john/geek_stuff/Data Set/IMDB_sentiment_analysis_data/aclImdb/test/pos'

train_pos = [each_file for each_file in os.listdir(train_pos_sample_dir) if os.path.isfile(os.path.join(train_pos_sample_dir, each_file))]
train_neg = [each_file for each_file in os.listdir(train_neg_sample_dir) if os.path.isfile(os.path.join(train_neg_sample_dir, each_file))]
test_pos = [each_file for each_file in os.listdir(test_pos_sample_dir) if os.path.isfile(os.path.join(test_pos_sample_dir, each_file))]
test_neg = [each_file for each_file in os.listdir(test_neg_sample_dir) if os.path.isfile(os.path.join(test_neg_sample_dir, each_file))]

file_list = train_pos + train_neg + test_pos + test_neg

idx_to_vocab = {}
vocab_to_idx = {}
len_vocab = 0

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
    html_cleaned_data = re.sub('[^A-Za-z ?.!]+', '', html_cleaned_data)

    # Break data into words and remove stop Words(or unnecessary words which doesn't have much influence on meaning)
    stop_words = set(stopwords.words('english'))
    tokenized_word = word_tokenize(html_cleaned_data)
    filtered_sentences = [w for w in tokenized_word if not w in stop_words]
    print(filtered_sentences)

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
        if val.lower() not in vocab_to_idx:
            vocab_to_idx[val] = len_vocab + indx
            idx_to_vocab[len_vocab + indx] = val
            len_vocab += 1

    # TODO: Check most frequent occurring words in

# Save data to pickle object
print("Length of Vocab", len_vocab)
print(vocab_to_idx)
print('Saving Dictionary')
dictionary = {'vocab_to_index': vocab_to_idx, 'index_to_vocab': idx_to_vocab}
with open('data/movie_vocab' + '.pkl', 'wb') as f:
    pickle.dump(dictionary, f)
