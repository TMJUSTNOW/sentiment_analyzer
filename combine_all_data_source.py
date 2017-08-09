import os
import json
import numpy as np
import html.parser
import re
import pickle
import time

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd
import collections


stop_words = ['see', 'get', 'watch', 'computer', 'comedydrama', 'room', 'channel', 'buck', 'western','prof', 'island', 'wow','junior', 'artist','uplifting', 'dazzling']

##Load All dictionary and combine them to one
with open('/home/janmejaya/sentiment_files/data/rotten_movie_vocab.pkl', 'rb') as f:
    data = pickle.load(f)

rotten_vocab_frequency_tuple = data['vocab_frequency_tuple']
print('Length of vocab tuple ', len(rotten_vocab_frequency_tuple))
print(rotten_vocab_frequency_tuple[:1000])

with open('/home/janmejaya/sentiment_files/data/movie_vocab.pkl', 'rb') as f:
    data = pickle.load(f)
imdb_vocab_frequency = data['vocab_frequency_tuple']
print('IMDB len of vocab tuple ', len(imdb_vocab_frequency))
print(imdb_vocab_frequency[24000:25000])

with open('/home/janmejaya/sentiment_files/data/twitter_vocab.pkl', 'rb') as f:
    data = pickle.load(f)
twitter_vocab_frequency = data['vocab_frequency_tuple']
print('Twitter len of vocab tuple ', len(twitter_vocab_frequency))
print(twitter_vocab_frequency[24000:25000])

complete_vocab = [(val,freq) for val,freq in rotten_vocab_frequency_tuple[:5000] if val not in stop_words]
word_list = list(list(zip(*complete_vocab))[0])

for val, freq in imdb_vocab_frequency[:25000]+twitter_vocab_frequency[:25000]:
    if val not in stop_words and val not in word_list:
        complete_vocab.append((val, freq))
        word_list.append(val)

print("Length of complete Vocab ", len(complete_vocab))
idx = 1
vocab_to_idx, idx_to_vocab = {}, {}
vocab_to_idx['<pad>'] = 0
idx_to_vocab[0] = '<pad>'
for val, freq in complete_vocab:
    vocab_to_idx[val] = idx
    idx_to_vocab[idx] = val
    idx += 1
print("len of complete vocab ", len(complete_vocab))
print("Max index ", idx)
print('Last word ', idx_to_vocab[idx-1])
print('Saving Dictionary')
dictionary = {'vocab_to_index': vocab_to_idx, 'index_to_vocab': idx_to_vocab, 'vocab_frequency_tuple': complete_vocab}
with open('/home/janmejaya/sentiment_files/data/complete_vocab_15_word2.pkl', 'wb') as f:
    pickle.dump(dictionary, f)

