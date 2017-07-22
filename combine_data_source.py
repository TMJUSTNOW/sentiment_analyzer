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


df = pd.read_csv('/home/john/geek_stuff/Data Set/Twitter Sentiment 3 class/training.1600000.processed.noemoticon.csv',
                                                                                                    encoding='latin-1')
df.columns = ['target', 'NR1', 'NR2', 'NR3', 'NR4', 'data']

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

word_lemmatizer = WordNetLemmatizer()

vocab_frequency = {}
len_vocab = 0
for data in df['data']:
    # Escape HTML char ir present
    html_parser = html.parser.HTMLParser()
    html_cleaned_data = html_parser.unescape(data)

    # Keep important punctuation
    html_cleaned_data = re.sub('[^A-Za-z ]+', '', html_cleaned_data)

    # Break data into words and remove stop Words(or unnecessary words which doesn't have much influence on meaning)
    stop_words = ['after', 'others', 'clud', 'would', 'id', 'self', 'youll', 'brother', 'second', 'shes',
                  'becomes', 'youd', 'etc', 'college', 'setting', 'ok', 'hair', 'wife', 'eye', 'thats', 'come', 'plot',
                  'english', 'so', 'house', 'year', 'men', 'be', 'go', 'line', 'lines', 'three', 'couple', 'next',
                  'home', 'else', 'saw', 'work', 'may', 'hes', 'day', 'goes', 'guy', 'theres', 'minutes', 'done',
                  'world', 'could', 'yourselves', 'us', 'actually', 'years', 'man', 'im', 'with', 'only', 'is', 'had',
                  'll', 'into', 'being', 'a', 'me', 'has', 'been', 'why', 'where', 'up', 'can', 'than', 'in', 'sex',
                  'days', 'from', 'who', 'the', 'should', 'your', 'themselves', 'while', 'we', 'against', 'further',
                  'these', 'those', 'were', 'be', 'out', 'him', 'our', 'which', 'just', 'doing', 'this', 'had',
                  'theirs', 'i', 'under', 'm', 'herself', 'by', 'at', 'until', 'here', 'were', 'if', 'myself', 'ma',
                  'ours', 'as', 'all', 'each', 'how', 'when', 'other', 'itself', 'an', 'my', 'did', 'them', 'ourselves',
                  'during', 'whom', 'am', 'o', 'same', 'are', 'have', 'above', 'what', 'both', 'do', 'off', 'before',
                  'or', 'having', 'now', 'too', 'on', 'and', 'through', 'she', 'his', 'do', 'there', 'won', 'they',
                  'to', 'are', 't', 'few', 'was', 'it', 'did', 'himself', 'her', 'such', 'have', 'yours', 'more', 'for',
                  're', 'will', 'ain', 's', 'you', 'their', 'about', 'between', 'that', 'once', 'does', 'shall', 've',
                  'he', 'of', 'y', 'own', 'again', 'd', 'any', 'does', 'was', 'its', 'below', 'hers', 'yourself']
    movie_stop_words = ['actors', 'actor', 'dvd', 'story', 'villain', 'acting', 'u', 'family', 'oh', 'cinematic',
                        'dialogue', 'hm', 'chynna', 'bogarde', 'shampoo', 'painfulbr', 'coolio', 'musclebound',
                        'baloney', 'hairline', 'joe', 'sexy', 'japanese', 'think', 'cinematography', 'father', 'say',
                        'studio', 'boy', 'direction', 'cinema', 'friend', 'writ', 'movies', 'american', 'actress',
                        'classic', 'star', 'directed', 'bor', 'title', 'human', 'enterta', 'ett', 'production',
                        'performances', 'films', 'characters', 'character', 'tv', 'main', 'sense', 'woman', 'girl',
                        'scenes', 'terest', 'scene', 'mak', 'director', 'ive', 'back', 'th', 'chemistry', 'bill', 'lee',
                        'audiences', 'producers', 'filmed', 'review', 'song', 'musical', 'songs', 'thriller', 'theater',
                        'br', 'dialog', 'james', 'one', 'first', 'four', 'five', 'ten', 'gore', 'ru', 'filmbr',
                        'viewers', 'dr', 'era', 'stefan', 'jrs', 'mar', 'palestinian', 'gandolfini', 'elisha', 'taboos',
                        'jessie', 'hindu', 'gandolfini', 'gandolfini', 'water', 'de', 'mov', 'moviebr', 'robert', 'v',
                        'liv', 'david', 'w', 'la', 'sitt', 'volv', 'tak', 'michael', 'ope', 'mr', 'hour', 'movie',
                        'music', 'film', 'two', 'th', 'youre', 'john', 'bor', 'ett', 'hollywood', 'drama', 'theyre',
                        'yeah', 'com', 'itbr', 'genre']

    # Performing Word Lemmatization on text
    lemmatized_data = []
    for word, typ in nltk.pos_tag(word_tokenize(html_cleaned_data)):
        typ = get_wordnet_pos(typ)
        if typ:
            lemmatized_data.append(word_lemmatizer.lemmatize(word, typ))
        else:
            lemmatized_data.append(word_lemmatizer.lemmatize(word))

    filtered_sentences = [w.strip(' ').lower() for w in lemmatized_data if
                          w.strip(' ').lower() not in stop_words + movie_stop_words]

    # create word to index and index to word dictionary
    vocab = set(filtered_sentences)
    for indx, val in enumerate(vocab):
        # Count frequency of the Word
        if val in vocab_frequency:
            vocab_frequency[val] += 1
        else:
            vocab_frequency[val] = 1
            len_vocab += 1

frequent_words_tuple = sorted(vocab_frequency.items(), key=lambda x: x[1], reverse=True)
frequent_words = [val for val, freq in frequent_words_tuple][:1000]
print(frequent_words)