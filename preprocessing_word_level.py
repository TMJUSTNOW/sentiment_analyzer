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

# Read data set(IMDB Review data)
train_pos_sample_dir = '/home/john/geek_stuff/Data Set/IMDB_sentiment_analysis_data/aclImdb/train/pos'
train_neg_sample_dir = '/home/john/geek_stuff/Data Set/IMDB_sentiment_analysis_data/aclImdb/train/neg'
test_neg_sample_dir = '/home/john/geek_stuff/Data Set/IMDB_sentiment_analysis_data/aclImdb/test/neg'
test_pos_sample_dir = '/home/john/geek_stuff/Data Set/IMDB_sentiment_analysis_data/aclImdb/test/pos'

train_pos = [each_file for each_file in os.listdir(train_pos_sample_dir) if os.path.isfile(os.path.join(train_pos_sample_dir, each_file))]
train_neg = [each_file for each_file in os.listdir(train_neg_sample_dir) if os.path.isfile(os.path.join(train_neg_sample_dir, each_file))]
test_pos = [each_file for each_file in os.listdir(test_pos_sample_dir) if os.path.isfile(os.path.join(test_pos_sample_dir, each_file))]
test_neg = [each_file for each_file in os.listdir(test_neg_sample_dir) if os.path.isfile(os.path.join(test_neg_sample_dir, each_file))]

file_list = test_pos[6250:] + test_neg[6250:]
print(len(file_list))
print(len(test_neg))
idx_to_vocab = {}
vocab_to_idx = {}
vocab_frequency = {}
len_vocab = 0
index_to_start = 1      # 0th index is used for padding

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

# for each_file in file_list:
#     try:
#         with open(os.path.join(train_pos_sample_dir, each_file), 'r') as f:
#             data = f.read()
#     except:
#         try:
#             with open(os.path.join(train_neg_sample_dir, each_file), 'r') as f:
#                 data = f.read()
#         except:
#             try:
#                 with open(os.path.join(test_neg_sample_dir, each_file), 'r') as f:
#                     data = f.read()
#             except:
#                 with open(os.path.join(test_pos_sample_dir, each_file), 'r') as f:
#                     data = f.read()
#
#     # PreProcess the data
#     # Escape HTML char ir present
#     html_parser = html.parser.HTMLParser()
#     html_cleaned_data = html_parser.unescape(data)
#
#
#     # Keep important punctuation
#     html_cleaned_data = re.sub('[^A-Za-z ]+', '', html_cleaned_data)
#
#
#     # Break data into words and remove stop Words(or unnecessary words which doesn't have much influence on meaning)
#     # TODO: There are few words like 'br', 'o', 'or' after listed to remove still appears in sentence
#     stop_words = ['after', 'others', 'clud', 'would', 'id', 'time', 'self', 'youll','brother', 'second', 'shes', 'becomes', 'youd','etc', 'college','setting', 'ok', 'hair','wife', 'eye','thats','come', 'plot', 'english', 'so', 'house','year','men', 'be', 'go', 'line', 'lines','three', 'couple', 'next', 'home','else', 'saw','work', 'may', 'hes', 'day', 'goes', 'guy','theres','minutes', 'done', 'world', 'could', 'yourselves', 'us', 'actually','years', 'man', 'im', 'with', 'only', 'is', 'had', 'll', 'into', 'being', 'a', 'me', 'has', 'been', 'why', 'where', 'up', 'can', 'than', 'in', 'sex', 'days','from', 'who', 'the', 'should', 'your', 'themselves', 'while', 'we', 'against', 'further', 'these', 'those', 'were', 'be', 'out', 'him', 'our', 'which', 'just', 'doing', 'this', 'had', 'theirs', 'i', 'under', 'm', 'herself', 'by', 'at', 'until', 'here', 'were', 'if', 'myself', 'ma', 'ours', 'as', 'all', 'each', 'how', 'when', 'other', 'itself', 'an', 'my', 'did', 'them', 'ourselves', 'during', 'whom', 'am', 'o', 'same', 'are', 'have', 'above', 'what', 'both', 'do', 'off', 'before', 'or', 'having', 'now', 'too', 'on', 'and', 'through', 'she', 'his', 'do', 'there', 'won', 'they', 'to', 'are', 't', 'few', 'was', 'it', 'did', 'himself', 'her', 'such', 'have', 'yours', 'more', 'for', 're', 'will', 'ain', 's', 'you', 'their', 'about', 'between', 'that', 'once', 'does', 'shall', 've', 'he', 'of', 'y', 'own', 'again', 'd', 'any', 'does', 'was', 'its', 'below', 'hers', 'yourself']
#     movie_stop_words = ['actors', 'actor', 'dvd',  'story', 'villain','acting', 'u', 'family', 'oh', 'cinematic','dialogue', 'hm', 'chynna', 'bogarde', 'shampoo', 'painfulbr', 'coolio', 'musclebound', 'baloney', 'hairline', 'joe', 'sexy','japanese','think', 'cinematography', 'father', 'say','studio', 'boy', 'direction', 'cinema', 'friend', 'writ','movies', 'american', 'actress','classic','star', 'directed', 'bor', 'title', 'human','enterta', 'ett', 'production', 'performances','films','characters', 'character', 'tv', 'main', 'sense','woman','girl', 'scenes', 'terest', 'scene', 'mak', 'director', 'ive', 'back', 'th','chemistry', 'bill', 'lee', 'audiences', 'producers', 'filmed', 'review', 'song', 'musical', 'songs', 'thriller', 'theater', 'br', 'dialog', 'james', 'one', 'first', 'four', 'five', 'ten', 'gore', 'ru', 'filmbr', 'viewers', 'dr', 'era','stefan', 'jrs', 'mar', 'palestinian', 'gandolfini', 'elisha', 'taboos', 'jessie', 'hindu', 'gandolfini', 'gandolfini', 'water', 'de', 'mov', 'moviebr', 'robert', 'v', 'liv', 'david', 'w', 'la', 'sitt', 'volv', 'tak', 'michael', 'ope', 'mr', 'hour', 'movie', 'music', 'film', 'two', 'th', 'youre', 'john', 'bor', 'ett', 'hollywood', 'drama', 'theyre', 'yeah', 'com', 'itbr', 'genre']
#
#     # Performing Word Lemmatization on text
#     lemmatized_data = []
#     for word, typ in nltk.pos_tag(word_tokenize(html_cleaned_data)):
#         typ = get_wordnet_pos(typ)
#         if typ:
#             lemmatized_data.append(word_lemmatizer.lemmatize(word, typ))
#         else:
#             lemmatized_data.append(word_lemmatizer.lemmatize(word))
#
#     filtered_sentences = [w.strip(' ').lower() for w in lemmatized_data if w.strip(' ').lower() not in stop_words+movie_stop_words]
#     if 'be' in filtered_sentences or 'o' in filtered_sentences or 'or' in filtered_sentences or 'br' in filtered_sentences:
#         print(lemmatized_data)
#         print(filtered_sentences)
#         print('\n\n')
#         time.sleep(10)
#
#
#
#     # TODO: Add low frequency words from classes
#     # create word to index and index to word dictionary
#     vocab = set(filtered_sentences)
#     for indx, val in enumerate(vocab):
#         # if val not in vocab_to_idx:
#         #     vocab_to_idx[val] = len_vocab + index_to_start
#         #     idx_to_vocab[len_vocab + index_to_start] = val
#         #     len_vocab += 1
#
#         # Count frequency of the Word
#         if val in vocab_frequency:
#             vocab_frequency[val] += 1
#         else:
#             vocab_frequency[val] = 1
#             len_vocab += 1
#     # TODO: Check most frequent occurring words in
#
# # Add Extra Padding words
# vocab_to_idx['<pad>'] = 0
# idx_to_vocab[0] = '<pad>'
# # Save data to pickle object
# print("Length of Vocab", len_vocab)
#
# frequent_words_tuple = sorted(vocab_frequency.items(), key=lambda x: x[1], reverse=True)
# frequent_words = [val for val, freq in frequent_words_tuple][:1000]
# print(frequent_words)
# # Before saving Dictionary create index of vocab based on descending order of their frequency
# # So if we are taking max 'n' words based on frequency then their index will be limited to 'n'
# idx = 1
# for val, freq in frequent_words_tuple:
#     vocab_to_idx[val] = idx
#     idx_to_vocab[idx] = val
#     idx += 1
# print('Saving Dictionary')
# dictionary = {'vocab_to_index': vocab_to_idx, 'index_to_vocab': idx_to_vocab, 'vocab_frequency_tuple': frequent_words_tuple}
# with open('data/movie_vocab.pkl', 'wb') as f:
#     pickle.dump(dictionary, f)



# Convert data in to index and store it
print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
with open('/home/john/sentiment_files/data/complete_vocab_15_word.pkl', 'rb') as f:
    data = pickle.load(f)
vocab_to_index = data['vocab_to_index']
index_to_vocab = data['index_to_vocab']
vocab_frequency_tuple = data['vocab_frequency_tuple']
vocab_len = len(data['vocab_to_index'])
print('Vocab len: ', vocab_len)
print("Len of vocab freq tuple ", len(vocab_frequency_tuple))

max_word = 15
max_features = 25000
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
    target_data = []
    input_data = np.array([], int)

    # Keep maximum frequent occurring words
    # frequent_words_tuple = sorted(vocab_frequency.items(), key=lambda x: x[1], reverse=True)[:max_features]
    frequent_words = [val for val, freq in vocab_frequency_tuple][:max_features]

    for index, each_file in enumerate(file_list):
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
                target_data.append(1)
        except:
            with open(os.path.join(neg_dir, each_file), 'r') as f:
                data = f.read()
                target_data.append(0)

        # Preprocess Data
        #Escape HTML char ir present
        html_parser = html.parser.HTMLParser()
        html_cleaned_data = html_parser.unescape(data)


        # Keep important punctuation
        html_cleaned_data = re.sub('[^A-Za-z ]+', '', html_cleaned_data)

        # Performing Word Lemmatization on text
        lemmatized_data = []
        for word, typ in nltk.pos_tag(word_tokenize(html_cleaned_data)):
            typ = get_wordnet_pos(typ)
            if typ:
                lemmatized_data.append(word_lemmatizer.lemmatize(word, typ))
            else:
                lemmatized_data.append(word_lemmatizer.lemmatize(word))

        truncated_review = []
        for each_word in lemmatized_data:
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
        if word_len:
            if word_len < max_word:
                truncated_review += [0] * (max_word - word_len)
            if len(input_data) != 0:
                input_data = np.vstack((input_data, truncated_review))
            else:
                input_data = np.hstack((input_data, truncated_review))
        else:
            # Remove latest added element from target data
            target_data.pop()
            print('Truncated Review ', truncated_review)
            print('Data ', data)
            print('Lemmatized data ', lemmatized_data)
    target_data = np.array(target_data)
    return (input_data, target_data)

# data_x, data_y = next_batch(len(file_list))
# train_dict = {'input': data_x, 'target': data_y}
# print('Input shape ', data_x.shape)
# print('Target shape ', data_y.shape)
# with open('/home/john/sentiment_files/data/complete_data_15_word/complete_train4.pkl', 'wb') as f:
#     pickle.dump(train_dict, f)

data_x, data_y = next_batch(len(file_list), test=True)
test_dict = {'input': data_x, 'target': data_y}
print('Max value in test: ', test_dict['input'].max(1).max())
print('Input shape ', data_x.shape)
print('Target shape ', data_y.shape)
with open('/home/john/sentiment_files/data/complete_data_15_word/complete_test.pkl', 'wb') as f:
    pickle.dump(test_dict, f)