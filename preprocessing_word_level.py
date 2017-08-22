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
import enchant

def get_wordnet_pos(treebank_tag):
    ## Removed Adjective from pos tagging as word_lemmatizer convert superlative degree to original form
    # if treebank_tag.startswith('J'):
    #     return wordnet.ADJ
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def data_preprocessing(data):
    # PreProcess the data
    # Escape HTML char ir present
    html_parser = html.parser.HTMLParser()
    html_cleaned_data = html_parser.unescape(data)

    # Remove all unnecessary special character
    html_cleaned_data = re.sub('[^A-Za-z ]+', '', html_cleaned_data)

    # Break data into words and remove stop Words(or unnecessary words which doesn't have much influence on meaning)
    # TODO: pickle and save stop word list
    stop_words = ['after', 'others', 'clud', 'would', 'id', 'time', 'self', 'youll', 'brother', 'second', 'shes',
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
    word_lemmatizer = WordNetLemmatizer()
    words_to_keep = []
    for word, typ in nltk.pos_tag(word_tokenize(html_cleaned_data)):
        typ = get_wordnet_pos(typ)
        if typ:
            lemmatized_word = word_lemmatizer.lemmatize(word, typ).lower()
        else:
            lemmatized_word = word_lemmatizer.lemmatize(word).lower()

        # Removing Stop words and correct spelled words
        # Remove all non-english or mis-spelled words
        enchant_dict = enchant.Dict("en_US")
        if enchant_dict.check(lemmatized_word) and lemmatized_word not in stop_words + movie_stop_words:
            words_to_keep.append(lemmatized_word)

    return words_to_keep

def create_vocab(file_to_read, file_dir_list, vocab_dir):

    len_vocab = 0
    vocab_frequency = {}
    vocab_to_idx = {}
    idx_to_vocab = {}
    for each_file in file_to_read:
        for each_dir in file_dir_list:
            try:
                with open(os.path.join(each_dir, each_file), 'r') as f:
                    data = f.read()
            except:
                continue

        preprocessed_word = data_preprocessing(data)

        # create word to index and index to word dictionary
        vocab = set(preprocessed_word)
        for indx, val in enumerate(vocab):
            # Count frequency of the Word
            if val in vocab_frequency:
                vocab_frequency[val] += 1
            else:
                vocab_frequency[val] = 1
                len_vocab += 1

    print("Length of Vocab", len_vocab)

    # Sort Vocab by their frequency
    frequent_words_tuple = sorted(vocab_frequency.items(), key=lambda x: x[1], reverse=True)

    # Before saving Dictionary create index of vocab based on descending order of their frequency
    # So if we are taking max 'n' words based on frequency then their index will be limited to 'n'
    # Add Extra Padding words
    vocab_to_idx['<pad>'] = 0
    idx_to_vocab[0] = '<pad>'
    idx = 1
    for val, freq in frequent_words_tuple:
        vocab_to_idx[val] = idx
        idx_to_vocab[idx] = val
        idx += 1
    print('Saving Dictionary')
    dictionary = {'vocab_to_index': vocab_to_idx, 'index_to_vocab': idx_to_vocab, 'vocab_frequency_tuple': frequent_words_tuple}
    with open(vocab_dir, 'wb') as f:
        pickle.dump(dictionary, f)
    print('Vocab successfully saved to: {0}'.format(vocab_dir))

def create_data(file_to_read, file_dir_list, vocab_dir, max_word, file_dir):
    with open(vocab_dir, 'rb') as f:
        data = pickle.load(f)
    vocab_to_index = data['vocab_to_index']
    index_to_vocab = data['index_to_vocab']
    vocab_frequency_tuple = data['vocab_frequency_tuple']
    vocab_len = len(data['vocab_frequency_tuple'])

    train_pos_sample_dir = file_dir_list[0]
    train_neg_sample_dir = file_dir_list[1]
    test_pos_sample_dir = file_dir_list[2]
    test_neg_sample_dir = file_dir_list[3]


    max_features = vocab_len
    target_data = []
    input_data = np.array([], int)

    # Keep maximum frequent occurring words
    frequent_words = [val for val, freq in vocab_frequency_tuple][:max_features]

    # Process each file
    for each_file in file_to_read:
        # Read Data from file
        try:
            with open(os.path.join(train_pos_sample_dir, each_file), 'r') as f:
                data = f.read()
                target_data.append(1)
        except:
            try:
                with open(os.path.join(train_neg_sample_dir, each_file), 'r') as f:
                    data = f.read()
                    target_data.append(0)
            except:
                try:
                    with open(os.path.join(test_neg_sample_dir, each_file), 'r') as f:
                        data = f.read()
                        target_data.append(0)
                except:
                    with open(os.path.join(test_pos_sample_dir, each_file), 'r') as f:
                        data = f.read()
                        target_data.append(1)

        # Pre-Process data word level
        preprocessed_word = data_preprocessing(data)

        # Create data
        truncated_review = []
        for each_word in preprocessed_word:
            # Take words which are frequent
            if each_word in frequent_words:
                try:
                    truncated_review.append(vocab_to_index[each_word])
                    if len(truncated_review) >= max_word:
                        break
                except Exception as exc:
                    print(
                        '[Exception] if Key word not present in vocab dict but present in frequent words its a bug: {0}'.format(
                            exc))

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

    # conver target to numpy array
    target_data = np.array(target_data)

    data_dict = {'input': input_data, 'target': target_data}
    with open(file_dir, 'wb') as f:
        pickle.dump(data_dict, f)

    print('Max value in test: ', data_dict['input'].max(1).max())
    print('Input shape ', data_dict['input'].shape)
    print('Target shape ', data_dict['target'].shape)
    print('Successfully saved data to {0}'.format(file_dir))

def _test_process_imdb_data():
    # Data Directory
    train_pos_sample_dir = '/home/john/geek_stuff/Data_Set/IMDB_sentiment_analysis_data/aclImdb/train/pos'
    train_neg_sample_dir = '/home/john/geek_stuff/Data_Set/IMDB_sentiment_analysis_data/aclImdb/train/neg'
    test_neg_sample_dir = '/home/john/geek_stuff/Data_Set/IMDB_sentiment_analysis_data/aclImdb/test/neg'
    test_pos_sample_dir = '/home/john/geek_stuff/Data_Set/IMDB_sentiment_analysis_data/aclImdb/test/pos'

    train_pos = [each_file for each_file in os.listdir(train_pos_sample_dir) if
                 os.path.isfile(os.path.join(train_pos_sample_dir, each_file))]
    train_neg = [each_file for each_file in os.listdir(train_neg_sample_dir) if
                 os.path.isfile(os.path.join(train_neg_sample_dir, each_file))]
    test_pos = [each_file for each_file in os.listdir(test_pos_sample_dir) if
                os.path.isfile(os.path.join(test_pos_sample_dir, each_file))]
    test_neg = [each_file for each_file in os.listdir(test_neg_sample_dir) if
                os.path.isfile(os.path.join(test_neg_sample_dir, each_file))]

    file_list = train_pos + train_neg + test_pos[:6250] + test_neg[:6250]
    vocab_dir = '/home/john/sentiment_files/data/movie_vocab.pkl'
    train_dir = '/home/janmejaya/sentiment_files/data/complete_data_aug9/imdb_train.pkl'
    test_dir = '/home/janmejaya/sentiment_files/data/complete_data_aug9/imdb_test.pkl'
    dir_list = [train_pos_sample_dir, train_neg_sample_dir, test_pos_sample_dir, test_neg_sample_dir]
    # creating Vocab from training data
    create_vocab(file_to_read=file_list, file_dir_list=dir_list, vocab_dir=vocab_dir)
    # Train data parameters
    max_word = 15           # maximum words to keep in a sentence
    create_data(file_to_read=file_list, file_dir_list=dir_list, vocab_dir=vocab_dir, max_word=max_word, file_dir=train_dir)

    # Test data
    file_list = test_pos[6250:] + test_neg[6250:]
    create_data(file_to_read=file_list, file_dir_list=dir_list, vocab_dir=vocab_dir, max_word=max_word,
                file_dir=test_dir)





if __name__ == '__main__':
    _test_process_imdb_data()