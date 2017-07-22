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


df = pd.DataFrame.from_csv('/home/janmejaya/Downloads/train.tsv', sep='\t', header=0)
print(df.columns)


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
for idx, data in enumerate(df['Phrase'].tolist()):
    if sentense_list[idx] in sentence_id:
        continue
    else:
        sentence_id.append(sentense_list[idx])
    if sentiment_list[idx] not in [0, 4]:
        continue

    # Escape HTML char ir present
    html_parser = html.parser.HTMLParser()
    html_cleaned_data = html_parser.unescape(data)

    # Keep important punctuation
    html_cleaned_data = re.sub('[^A-Za-z ]+', '', html_cleaned_data)

    # Break data into words and remove stop Words(or unnecessary words which doesn't have much influence on meaning)
    stop_words = ['after', 'others', 'clud', 'would', 'id', 'self', 'youll', 'brother', 'second', 'shes',
                  'becomes', 'youd', 'etc', 'college','shin','server', 'uni', 'jordanknight','wishing', 'swine', 'adam', 'prom','england','bath', 'battery', 'button', 'plz', 'freakin', 'click', 'addict', 'topic', 'code', 'tough', 'island', 'wave', 'official', 'oops', 'setting', 'pls', 'ok','thunder', 'oo', 'grandma', 'fave', 'size', 'oooh', 'charge', 'disney', 'guys', 'xbox', 'hair', 'wife', 'eye', 'thats', 'come', 'plot',
                  'english', 'so', 'house', 'year', 'men', 'cooky','gunna', 'wall', 'tree', 'client', 'essay', 'sushi','germany', 'twice', 'be','jon', 'go','loud','fav', 'danny', 'boat', 'style', 'girlfriend','vid', 'spring', 'jam', 'soup', 'line', 'lines', 'direct', 'three', 'couple', 'next',
                  'home', 'else', 'saw', 'work', 'may','bottle', 'machine', 'sunburn','mmm', 'grad','mmm', 'available', 'followers', 'chris', 'joy', 'teach', 'kitty', 'kate', 'mcfly', 'demi', 'france', 'davidarchie', 'tip', 'gutted', 'ima', 'present', 'airport','babe', 'uh', 'attack', 'woohoo', 'mama', 'local', 'jack', 'student', 'smoke', 'brazil', 'york', 'taking', 'couch', 'mitchelmusso', 'file', 'butt', 'alive', 'imma', 'personal', 'kitchen', 'hes','west', 'push', 'king', 'training', 'ate', 'boys', 'wii', 'strange', 'taylorswift', 'switch', 'mo', 'made', 'canada', 'jb', 'mornin', 'fry', 'day', 'channel','bird', 'husband','blackberry', 'gig', 'goes', 'fam', 'guy', 'theres', 'minutes', 'done',
                  'world', 'could', 'yourselves', 'text','august', 'energy', 'ad','egg', 'tonite', 'finale', 'assignment', 'spell', 'science', 'social', 'treat', 'us', 'weight', 'level', 'exams', 'million', 'america', 'diet', 'getting','toe', 'stage', 'dammit','yum','taylor', 'gay','sooooo', 'daddy', 'eating', 'dang', 'awwww', 'nyc', 'paint', 'album', 'ny', 'jon', 'series', 'store','actually', 'years', 'man', 'im', 'with', 'only', 'is', 'had',
                  'll', 'into', 'being', 'a', 'ohhh', 'me', 'has', 'mine', 'ask', 'dinner', 'been', 'why', 'where', 'up', 'can', 'than', 'in', 'sex',
                  'days', 'from', 'who', 'the', 'should', 'your','blah', 'thanx', 'weve', 'xo', 'hannah', 'jonathanrknight', 'bf', 'mobile', 'tweeps', 'vegas', 'teacher', 'realise','kno','shopping','area', 'outta', 'bother', 'dentist', 'session','pack', 'themselves', 'chicago', 'donniewahlberg','while', 'we', 'against', 'further',
                  'these', 'those', 'were', 'be', 'out','holy', 'station', 'midnight', 'him','cuz', 'teeth', 'bummer', 'office', 'argh', 'our', 'which', 'yo', 'jus', 'piss', 'daughter', 'cup', 'wit', 'just', 'box', 'note', 'tom', 'myspace', 'service', 'cheer', 'raining', 'road', 'rainy', 'window', 'tummy','doing', 'this', 'had',
                  'theirs', 'i', 'under', 'm', 'herself','show', 'club', 'alone','bitch', 'interview', 'yummy', 'type', 'bbq', 'apple', 'graduation', 'by', 'at', 'until', 'here','officially', 'case', 'idk', 'bike', 'itll', 'garden', 'sims', 'leg', 'boyfriend', 'camera', 'jonasbrothers', 'son', 'were', 'if', 'myself', 'ma',
                  'ours', 'as', 'all', 'each', 'how', 'when', 'thanks', 'other', 'itself', 'an','thursday', 'ahhh', 'miley', 'my', 'did', 'them', 'ourselves',
                  'during', 'whom', 'am', 'o', 'same', 'are', 'number', 'jonas', 'bc','french', 'business', 'yup', 'tuesday', 'have','ipod', 'above', 'what', 'both', 'do', 'off', 'before',
                  'or', 'having', 'now', 'too', 'on', 'and', 'town', 'cd', 'company', 'xoxo', 'doctor', 'wedding', 'dm', 'wednesday', 'twilight', 'luv', 'notice', 'ps', 'smell', 'front', 'hubby', 'shot', 'through', 'ppl', 'mood','meeting', 'lil', 'she', 'his', 'do', 'bday', 'laptop', 'youtube', 'church', 'side','mother','homework', 'goin', 'tour', 'lady', 'there', 'won', 'they',
                  'to', 'are', 't', 'few', 'was','woo', 'gym', 'shirt', 'wed', 'ache', 'si', 'chicken', 'goodbye', 'yr', 'g', 'pizza', 'talk', 'today', 'hows', 'c','mum', 'episode', 'sleepy', 'hr', 'via', 'xxx','london', 'ahh', 'chocolate','phone', 'date','red','city', 'soooo', 'nite', 'xd', 'ice','bout','it', 'did', 'june','cat', 'afternoon', 'tommcfly', 'hop', 'lmao', 'himself','facebook', 'aw', 'fix', 'asleep', 'her', 'such', 'have', 'yours', 'more', 'for',
                  're', 'will', 'ain', 's', 'you', 'l', 'pink', 'search', 'gettin','ouch', 'bum', 'cousin', 'uk','stomach', 'shoe', 'google', 'their', 'yall', 'fb','team', 'flu', 'thx', 'wtf', 'voice', 'annoy', 'finger', 'mtv', 'math', 'followfriday', 'yep', 'download', 'bye', 'rain', 'about', 'between', 'wan', 'hug', 'season','tweet', 'hey','weekend', 'that', 'once', 'does', 'shall', 've',
                  'he', 'of', 'y', 'own', 'again', 'd', 'sun','due', 'em', 'mac', 'page', 'yea', 'message', 'any', 'game', 'hehe', 'st', 'train', 'til', 'ticket','pic', 'study','damn','morning', 'tonight', 'thing','twitter', 'haha', 'tomorrow', 'does', 'was', 'its', 'below', 'hers', 'yourself']
    movie_stop_words = ['actors', 'actor', 'dvd', 'omg', 'wow', 'exam', 'cold','okay','parent', 'k', 'e', 'project', 'hmm', 'website', 'da', 'hat', 'youve', 'tho', 'weather', 'hi', 'car', 'post', 'friday', 'kid', 'sunday', 'party','read', 'story', 'x','ur', 'lol', 'yay', 'night', 'na', 'gon', 'villain', 'acting', 'u', 'family', 'oh', 'cinematic',
                        'dialogue', 'hm', 'chynna', 'bogarde', 'ya', 'photo', 'ah', 'place', 'picture','ddlovato','p', 'fuck', 'shit', 'plan', 'iphone', 'video', 'month', 'aww','ta', 'n', 'yesterday', 'mom', 'ugh',  'shampoo', 'painfulbr', 'coolio', 'musclebound',
                        'baloney', 'hairline', 'joe','blog', 'email', 'mileycyrus','breakfast', 'dude', 'shower', 'dance', 'visit', 'wear', 'site', 'pick', 'ha', 'concert', 'online', 'sexy', 'soo','sister', 'internet', 'boo', 'btw', 'news', 'japanese', 'r', 'name','drink','b', 'hahaha', 'monday', 'think', 'cinematography', 'father', 'say',
                        'studio', 'boy', 'direction', 'cinema', 'pm', 'friend', 'word', 'awww', 'goodnight', 'trip', 'writ', 'movies', 'american', 'actress',
                        'classic', 'star', 'directed', 'bor', 'title', 'coffee', 'sooo', 'kinda', 'beach', 'dog', 'saturday','xx', 'dad', 'human', 'enterta', 'ett', 'production',
                        'performances', 'films', 'characters', 'character', 'tv', 'main', 'sense', 'woman', 'girl',
                        'scenes', 'terest', 'scene', 'mak', 'director', 'ive', 'back', 'th', 'chemistry', 'bill', 'lee',
                        'audiences', 'producers', 'filmed', 'review', 'song', 'musical', 'songs', 'thriller', 'theater',
                        'br', 'dialog', 'james', 'one', 'first', 'four', 'five', 'ten', 'gore', 'ru', 'filmbr',
                        'viewers', 'dr', 'era', 'stefan', 'jrs', 'mar', 'palestinian', 'gandolfini', 'elisha', 'taboos',
                        'jessie', 'hindu', 'gandolfini', 'gandolfini', 'water', 'de', 'mov', 'moviebr', 'robert', 'v',
                        'liv', 'david', 'w', 'la', 'sitt', 'volv', 'tak', 'michael', 'ope', 'mr', 'hour', 'movie',
                        'music', 'film', 'two','nt', 'th', 'rrb', 'lrb','youre', 'john', 'bor', 'ett', 'hollywood', 'drama', 'theyre',
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
print(frequent_words_tuple[:1000])