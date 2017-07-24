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


df = pd.DataFrame.from_csv('/home/janmejaya/Downloads/train.tsv', sep='\t', header=0)
print(df.columns)

# Break data into words and remove stop Words(or unnecessary words which doesn't have much influence on meaning)
stop_words = ['after', 'others', 'clud', 'would', 'id', 'self', 'youll', 'brother', 'second', 'shes',
              'becomes', 'youd', 'etc', 'college', 'shin', 'server', 'uni', 'jordanknight', 'wishing', 'swine', 'adam',
              'prom', 'england', 'bath', 'battery', 'button', 'plz', 'freakin', 'click', 'addict', 'topic', 'code',
              'tough', 'island', 'wave', 'official', 'oops', 'setting', 'pls', 'ok', 'thunder', 'oo', 'grandma', 'fave',
              'size', 'oooh', 'charge', 'disney', 'guys', 'xbox', 'hair', 'wife', 'eye', 'thats', 'come', 'plot',
              'english', 'so', 'house', 'year', 'way', 'men', 'take', 'cooky', 'gunna', 'wall', 'tree', 'client',
              'essay', 'sushi', 'germany', 'twice', 'be', 'jon', 'go', 'loud', 'fav', 'danny', 'boat', 'style',
              'girlfriend', 'vid', 'spring', 'jam', 'soup', 'line', 'lines', 'direct', 'three', 'couple', 'next',
              'home', 'else', 'saw', 'work', 'may', 'bottle', 'machine', 'sunburn', 'mmm', 'grad', 'mmm', 'available',
              'followers', 'chris', 'joy', 'teach', 'kitty', 'kate', 'mcfly', 'demi', 'france', 'davidarchie', 'tip',
              'gutted', 'ima', 'present', 'airport', 'babe', 'uh', 'attack', 'woohoo', 'mama', 'local', 'jack',
              'student', 'smoke', 'brazil', 'york', 'taking', 'couch', 'mitchelmusso', 'file', 'butt', 'alive', 'imma',
              'personal', 'kitchen', 'hes', 'west', 'push', 'king', 'training', 'ate', 'boys', 'wii', 'strange',
              'taylorswift', 'switch', 'mo', 'made', 'canada', 'jb', 'mornin', 'fry', 'day', 'channel', 'bird',
              'husband', 'blackberry', 'gig', 'goes', 'fam', 'guy', 'theres', 'minutes', 'done',
              'world', 'could', 'yourselves', 'text', 'august', 'energy', 'ad', 'egg', 'tonite', 'narrative', 'point',
              'everyone', 'sequence', 'deliver', 'finale', 'sit', 'assignment', 'spell', 'science', 'social', 'treat',
              'us', 'weight', 'level', 'exams', 'million', 'america', 'diet', 'getting', 'toe', 'stage', 'dammit',
              'yum', 'taylor', 'gay', 'sooooo', 'daddy', 'eating', 'dang', 'awwww', 'nyc', 'paint', 'album', 'ny',
              'jon', 'series', 'store', 'actually', 'years', 'man', 'im', 'with', 'only', 'is', 'had',
              'll', 'into', 'being', 'a', 'ohhh', 'me', 'seem', 'has', 'mine', 'ask', 'age', 'dinner', 'been', 'why',
              'where', 'up', 'can', 'than', 'in', 'sex',
              'days', 'from', 'who', 'the', 'should', 'your', 'blah', 'everything', 'thanx', 'weve', 'xo', 'hannah',
              'jonathanrknight', 'bf', 'mobile', 'tweeps', 'vegas', 'teacher', 'realise', 'kno', 'shopping', 'area',
              'outta', 'bother', 'dentist', 'session', 'pack', 'themselves', 'chicago', 'donniewahlberg', 'while', 'we',
              'against', 'further',
              'these', 'those', 'were', 'be', 'out', 'holy', 'turn', 'leave', 'station', 'midnight', 'him', 'cuz',
              'teeth', 'bummer', 'office', 'argh', 'our', 'which', 'yo', 'jus', 'piss', 'daughter', 'cup', 'wit',
              'just', 'box', 'note', 'tom', 'myspace', 'service', 'cheer', 'raining', 'road', 'rainy', 'window',
              'tummy', 'doing', 'this', 'had',
              'theirs', 'i', 'under', 'm', 'herself', 'show', 'club', 'alone', 'bitch', 'cheap', 'interview', 'yummy',
              'type', 'bbq', 'apple', 'graduation', 'by', 'at', 'until', 'here', 'officially', 'case', 'idk', 'bike',
              'itll', 'garden', 'sims', 'leg', 'boyfriend', 'camera', 'jonasbrothers', 'son', 'were', 'if', 'myself',
              'ma', 'delivery',
              'ours', 'as', 'all', 'each', 'how', 'when', 'adult', 'thanks', 'whose', 'other', 'itself', 'an',
              'thursday', 'ahhh', 'miley', 'my', 'did', 'them', 'ourselves',
              'during', 'whom', 'am', 'o', 'same', 'are', 'visual', 'maintain', 'number', 'wannabe', 'jonas', 'bc',
              'french', 'business', 'yup', 'tuesday', 'have', 'ipod', 'above', 'what', 'both', 'do', 'off', 'before',
              'or', 'having', 'now', 'too', 'on', 'and', 'town', 'ms', 'cd', 'ode', 'imagery', 'company', 'xoxo',
              'doctor', 'wedding', 'dm', 'wednesday', 'twilight', 'luv', 'notice', 'ps', 'smell', 'front', 'hubby',
              'shot', 'through', 'ppl', 'mood', 'meeting', 'lil', 'she', 'his', 'do', 'bday', 'laptop', 'youtube',
              'church', 'side', 'mother', 'homework', 'goin', 'tour', 'lady', 'there', 'won', 'they',
              'to', 'are', 't', 'few', 'put', 'was', 'woo', 'within', 'gang', 'wellacted', 'stir', 'bring', 'soap',
              'gym', 'shirt', 'wed', 'ache', 'si', 'chicken', 'goodbye', 'yr', 'g', 'pizza', 'talk', 'today', 'hows',
              'c', 'mum', 'episode', 'sleepy', 'hr', 'via', 'xxx', 'london', 'ahh', 'chocolate', 'phone', 'date', 'red',
              'city', 'soooo', 'nite', 'xd', 'ice', 'bout', 'it', 'did', 'june', 'cat', 'afternoon', 'tommcfly', 'hop',
              'lmao', 'himself', 'facebook', 'aw', 'fix', 'asleep', 'her', 'such', 'have', 'yours', 'more', 'for',
              're', 'will', 'ain', 's', 'you', 'identity', 'leaden', 'christmas', 'americans', 'street', 'preposterous',
              'l', 'timing', 'pink', 'audience', 'viewer', 'search', 'get', 'toward', 'gettin', 'ouch', 'bum', 'cousin',
              'uk', 'stomach', 'shoe', 'google', 'their', 'yall', 'fb', 'team', 'flu', 'thx', 'wtf', 'voice', 'annoy',
              'finger', 'mtv', 'math', 'followfriday', 'yep', 'download', 'bye', 'rain', 'about', 'between', 'wan',
              'hug', 'season', 'tweet', 'hey', 'weekend', 'that', 'once', 'does', 'shall', 've',
              'he', 'of', 'y', 'own', 'again', 'roll', 'international', 'medium', 'martin', 'make', 'd', 'people',
              'amount', 'screen', 'sun', 'due', 'em', 'mac', 'page', 'yea', 'message', 'any', 'game', 'hehe', 'st',
              'train', 'til', 'ticket', 'pic', 'study', 'damn', 'morning', 'tonight', 'thing', 'twitter', 'haha',
              'tomorrow', 'does', 'was', 'its', 'below', 'hers', 'yourself']
movie_stop_words = ['actors', 'actor', 'idea', 'ca', 'provide', 'old', 'timeless', 'cultural', 'collection', 'target',
                    'return', 'want', 'example', 'filmmaker', 'entertainment', 'touch', 'feature', 'use', 'care', 'dvd',
                    'omg', 'wow', 'minute', 'heart', 'set', 'experience', 'piece', 'act', 'also', 'script', 'action',
                    'end', 'cast', 'documentary', 'see', 'life', 'exam', 'cold', 'okay', 'parent', 'k', 'e', 'project',
                    'hmm', 'website', 'da', 'hat', 'youve', 'tho', 'weather', 'hi', 'car', 'post', 'friday', 'kid',
                    'sunday', 'party', 'read', 'story', 'x', 'ur', 'lol', 'yay', 'night', 'na', 'gon', 'villain',
                    'acting', 'u', 'family', 'oh', 'cinematic',
                    'dialogue', 'hm', 'chynna', 'screenplay', 'head', 'popcorn', 'goofy', 'drive', 'simpleminded',
                    'seat', 'part', 'event', 'nature', 'sustain', 'routine', 'sentimental', 'step', 'instantly',
                    'political', 'raise', 'thoroughly', 'engage', 'summer', 'sequel', 'bogarde', 'moment', 'history',
                    'ya', 'thoughtful', 'become', 'photo', 'fan', 'keep', 'ah', 'place', 'write', 'filmmaking',
                    'subject', 'art', 'worth', 'young', 'picture', 'ddlovato', 'p', 'fuck', 'shit', 'plan', 'iphone',
                    'video', 'month', 'aww', 'ta', 'n', 'yesterday', 'mom', 'ugh', 'shampoo', 'painfulbr', 'coolio',
                    'musclebound',
                    'baloney', 'hairline', 'joe', 'blog', 'belly', 'aweinspiring', 'power', 'capture', 'herzog',
                    'lawrence', 'et', 'opera', 'product', 'score', 'email', 'mileycyrus', 'breakfast', 'dude', 'effect',
                    'shower', 'dance', 'visit', 'wear', 'site', 'pick', 'ha', 'concert', 'online', 'sexy', 'soo',
                    'sister', 'internet', 'boo', 'btw', 'news', 'japanese', 'r', 'name', 'drink', 'b', 'hahaha',
                    'monday', 'think', 'cinematography', 'father', 'say',
                    'studio', 'boy', 'direction', 'remind', 'toilet', 'painting', 'cinema', 'evening', 'soundtrack',
                    'design', 'dreadful', 'pm', 'class', 'mouth', 'stoop', 'occasionally', 'animal', 'friend', 'word',
                    'awww', 'imax', 'conviction', 'goodnight', 'trip', 'writ', 'movies', 'american', 'actress',
                    'classic', 'star', 'directed', 'spiritual', 'crime', 'wry', 'final', 'inane', 'directorial',
                    'build', 'fashion', 'provocative', 'add', 'moviemaking', 'junk', 'gem', 'stretch', 'scream',
                    'harvard', 'crowd', 'simultaneously', 'refreshingly', 'future', 'sight', 'doze', 'open', 'result',
                    'heavyhanded', 'offbeat', 'touching', 'visuals', 'dream', 'material', 'bor', 'emerge', 'title',
                    'coffee', 'sooo', 'kinda', 'beach', 'dog', 'saturday', 'xx', 'dad', 'human', 'enterta', 'ett',
                    'production',
                    'performances', 'wish', 'rise', 'feces', 'base', 'examination', 'felt', 'deal', 'flimsy',
                    'demonstrate', 'kill', 'seek', 'air', 'share', 'totally', 'execute', 'films', 'face', 'firstrate',
                    'allen', 'skip', 'element', 'lifeless', 'detailed', 'franchise', 'combine', 'quiet', 'inside',
                    'treasure', 'pop', 'landscape', 'usually', 'superficial', 'standard', 'dramatically', 'characters',
                    'writerdirector', 'oldfashioned', 'help', 'character', 'oscar', 'adaptation', 'tv', 'main', 'sense',
                    'woman', 'girl', 'middle',
                    'scenes', 'terest', 'substance', 'reach', 'newcomer', 'graphic', 'country', 'moviegoing', 'vision',
                    'toss', 'guess', 'space', 'aspect', 'crass', 'someone', 'school', 'witless', 'excuse', 'meander',
                    'equally', 'merely', 'animated', 'hear', 'person', 'scene', 'suck', 'cheesy', 'creativity',
                    'protagonist', 'artificial', 'rest', 'term', 'ago', 'throughout', 'scifi', 'familiar', 'technical',
                    'manner', 'amateurish', 'brown', 'book', 'rhythm', 'exactly', 'mak', 'cut', 'artist', 'image',
                    'edge', 'question', 'director', 'ive', 'back', 'th', 'chemistry', 'bill', 'lee',
                    'audiences', 'producers', 'spielberg', 'mindnumbingly', 'bond', 'document', 'strike', 'ii',
                    'washington', 'poetry', 'soar', 'vital', 'frame', 'remake', 'filmed', 'ensemble', 'watchable',
                    'wellmade', 'sign', 'wo', 'sound', 'walk', 'form', 'shoot', 'inept', 'melodrama', 'view',
                    'animation', 'review', 'song', 'musical', 'songs', 'black', 'though', 'thriller', 'theater',
                    'br', 'dialog', 'james', 'one', 'clarity', 'whimsical', 'odd', 'tear', 'expect', 'guarantee',
                    'journey', 'dead', 'stunt', 'maker', 'novel', 'choose', 'lift', 'period', 'sensual', 'portrayal',
                    'thumb', 'british', 'body', 'quickly', 'characterization', 'worthwhile', 'hand', 'storyline',
                    'first', 'four', 'five', 'ten', 'master', 'wait', 'tone', 'gore', 'ru', 'filmbr',
                    'viewers', 'dr', 'debut', 'predictable', 'fit', 'drug', 'goodhearted', 'jackson', 'reality',
                    'tired', 'career', 'chase', 'suspenseful', 'usual', 'consistently', 'derivative', 'dub', 'steven',
                    'transform', 'culture', 'release', 'death', 'money', 'spy', 'insight', 'motion', 'begin', 'era',
                    'stefan', 'jrs', 'mar', 'palestinian', 'gandolfini', 'elisha', 'taboos',
                    'jessie', 'hindu', 'let', 'gag', 'heartwarming', 'carry', 'value', 'thoughtprovoking', 'gandolfini',
                    'gandolfini', 'theme', 'job', 'die', 'relationship', 'core', 'past', 'modern', 'heartfelt',
                    'version', 'live', 'pay', 'water', 'de', 'mov', 'moviebr', 'robert', 'v',
                    'liv', 'david', 'w', 'la', 'sitt', 'volv', 'tak', 'michael', 'ope', 'mr', 'hour', 'movie',
                    'music', 'film', 'two', 'nt', 'th', 'rrb', 'lrb', 'youre', 'john', 'bor', 'ett', 'hollywood',
                    'drama', 'theyre', 'yeah', 'com', 'itbr', 'genre']


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
sentense_list = df['SentenceId'].tolist()
sentiment_list = df['Sentiment'].tolist()
sentence_id = []
# for idx, data in enumerate(df['Phrase'].tolist()):
#     if sentiment_list[idx] not in [0, 4]:
#         continue
#
#     # Escape HTML char ir present
#     html_parser = html.parser.HTMLParser()
#     html_cleaned_data = html_parser.unescape(data)
#
#     # Keep important punctuation
#     html_cleaned_data = re.sub('[^A-Za-z ]+', '', html_cleaned_data)
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
#     filtered_sentences = [w.strip(' ').lower() for w in lemmatized_data if
#                           w.strip(' ').lower() not in stop_words + movie_stop_words]
#
#     # create word to index and index to word dictionary
#     vocab = set(filtered_sentences)
#     counter += 1
#     for indx, val in enumerate(vocab):
#         # Count frequency of the Word
#         if val in vocab_frequency:
#             vocab_frequency[val] += 1
#         else:
#             vocab_frequency[val] = 1
#             len_vocab += 1
#
# frequent_words_tuple = sorted(vocab_frequency.items(), key=lambda x: x[1], reverse=True)
# frequent_words = [val for val, freq in frequent_words_tuple][:1000]
# print(frequent_words)
# print(frequent_words_tuple[:1000])
# print('Length of vocab: ', counter)
# # print(collections.Counter(word_len))
# idx = 1
# vocab_to_idx, idx_to_vocab = {}, {}
# vocab_to_idx['<pad>'] = 0
# idx_to_vocab[0] = '<pad>'
# for val, freq in frequent_words_tuple:
#     vocab_to_idx[val] = idx
#     idx_to_vocab[idx] = val
#     idx += 1
# print('Saving Dictionary')
# dictionary = {'vocab_to_index': vocab_to_idx, 'index_to_vocab': idx_to_vocab, 'vocab_frequency_tuple': frequent_words_tuple}
# with open('/home/janmejaya/sentiment_files/data/rotten_movie_vocab.pkl', 'wb') as f:
#     pickle.dump(dictionary, f)

# Convert data in to index and store it
print('Loading data...')
with open('/home/janmejaya/sentiment_files/data/rotten_movie_vocab.pkl', 'rb') as f:
    data = pickle.load(f)
vocab_to_index = data['vocab_to_index']
index_to_vocab = data['index_to_vocab']
vocab_frequency_tuple = data['vocab_frequency_tuple']
vocab_len = len(data['vocab_to_index'])
print('Vocab len: ', vocab_len)

max_word = 10
max_features = 1000
batch_size = 16278
target_data = np.zeros([batch_size], dtype=np.int32)
input_data = np.zeros([batch_size, max_word], dtype=np.int32)
counter = 0
for idx, data in enumerate(df['Phrase'].tolist()):
    if sentiment_list[idx] not in [0, 4]:
        continue
    # Escape HTML char ir present
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

    frequent_words = [val for val, freq in vocab_frequency_tuple][:max_features]

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
                print(
                    '[Exception] if Key word not present in vocab dict but present in frequent words its a bug: {0}'.format(
                        exc))

    # Pad appropriately if less words are present
    word_len = len(truncated_review)
    if word_len < max_word:
        truncated_review += [0] * (max_word - word_len)
    input_data[counter] = truncated_review
    if sentiment_list[idx] == 4:
        target_data[counter] = 1
    else:
        target_data[counter] = 0
    counter += 1

print(input_data.max(1).max())
stri = ''
ind = batch_size-1
print('Data\n', input_data[ind])
for indx in input_data[ind]:
    stri += ' ' + index_to_vocab[indx]
print(stri)
print('Traget: ', target_data[ind])

train_dict = {'input': input_data, 'target': target_data}

with open('/home/janmejaya/sentiment_files/data/rotten_movie_train.pkl', 'wb') as f:
    pickle.dump(train_dict, f)