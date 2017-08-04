import pickle
import time
import re

from line_profiler import LineProfiler
from keras.models import model_from_json
from nltk.tokenize import word_tokenize
import numpy as np
import tensorflow as tf
from nltk.corpus import wordnet
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.keyedvectors import KeyedVectors

##### Analyze these sntence
#indian bank write percent value bad loan large stress asset account hit billion rating agency
# indian benchmark gain most over week lead company
# lg electronics inc thursday continued loss mobile unit limited growth profit percent firm prepare release
# stock jump nearly percent optical maker report rise percent rs period end june drive strong

word_vectors = KeyedVectors.load_word2vec_format(
        '/home/john/geek_stuff/Data_Set/Google_News_corpus/GoogleNews-vectors-negative300.bin', binary=True)

def get_wordnet_pos(treebank_tag):

    ## Removed Adjective from pos tagging as word_lemmatizer convert superlative degree to original form
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

def predict_sentiment(clean_string):

    ##### Code for With out using Word2vec

    # vocab_to_index, index_to_vocab, vocab_frequency_tuple = load_vocab()
    # imdb_model = attach_model()
    # # Model Parameters
    # max_word = 15
    # max_features = 25000
    # # evaluate loaded model on test data
    # imdb_model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    # frequent_words = [val for val, freq in vocab_frequency_tuple][:max_features]
    # word_lemmatizer = WordNetLemmatizer()
    # clean_string = re.sub('[^A-Za-z ]+', '', clean_string)
    # input_data = np.zeros([1, max_word], dtype=np.int32)
    # lemmatized_data = []
    # for word, typ in nltk.pos_tag(word_tokenize(clean_string)):
    #     typ = get_wordnet_pos(typ)
    #     if typ:
    #         lemmatized_data.append(word_lemmatizer.lemmatize(word, typ))
    #     else:
    #         lemmatized_data.append(word_lemmatizer.lemmatize(word))
    # truncated_data = []
    # for each_word in list(map(str.lower, lemmatized_data)):
    #     if each_word in frequent_words:
    #         truncated_data.append(vocab_to_index[each_word])
    #         if len(truncated_data) >= max_word:
    #             break
    # # Perform Extra Padding
    # if len(truncated_data) < max_word:
    #     truncated_data += [0] * (max_word - len(truncated_data))
    #
    # input_data[0] = truncated_data
    # prediction = imdb_model.predict(input_data)
    # sess = tf.Session()
    # score = sess.run(tf.nn.softmax(prediction))
    # prediction = np.argmax(score, axis=1)
    # stri = ''
    # for idx in input_data[0]:
    #     if idx == 0:
    #         continue
    #     stri += ' ' + index_to_vocab[idx]
    #
    # print('With a score of -Ve: {0}% +Ve: {1}%'.format(int(score[0][0]*100), int(score[0][1]*100)))
    # if abs(score[0][0] - score[0][1]) <= 0.15:
    #     print('Detected Sentiment Neutral')
    # else:
    #     prediction = np.argmax(score, axis=1)
    #     if prediction == 0:
    #         print('Sentiment Detected Negative')
    #     elif prediction == 1:
    #         print('Sentiment Detected Positive')
    # print('Words Used for Prediction:  {0}'.format(stri))

    #### With googles Word2vec
    imdb_model = attach_model()
    max_word = 15

    imdb_model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    word_lemmatizer = WordNetLemmatizer()
    clean_string = re.sub('[^A-Za-z ]+', '', clean_string)
    lemmatized_data = []
    for word, typ in nltk.pos_tag(word_tokenize(clean_string)):
        typ = get_wordnet_pos(typ)
        if typ:
            lemmatized_data.append(word_lemmatizer.lemmatize(word, typ))
        else:
            lemmatized_data.append(word_lemmatizer.lemmatize(word))
    truncated_data = []
    # word_vectors = KeyedVectors.load_word2vec_format(
    #     '/home/john/geek_stuff/Data_Set/Google_News_corpus/GoogleNews-vectors-negative300.bin', binary=True)

    stri = ''
    input_data = np.zeros([1, max_word, 300], dtype=np.float64)
    idx = 0
    for each_word in list(map(str.lower, lemmatized_data)):
        try:
            input_data[0, idx] = word_vectors[each_word]
            stri += ' ' + each_word
            idx += 1
        except Exception as exc:
            continue
        if idx >= max_word:
            break


    prediction = imdb_model.predict(input_data)
    sess = tf.Session()
    print('Prediction ', prediction)
    score = sess.run(tf.nn.softmax(prediction))
    prediction = np.argmax(score, axis=1)

    print('With a score of -Ve: {0}% +Ve: {1}%'.format(int(score[0][0]*100), int(score[0][1]*100)))
    if abs(score[0][0] - score[0][1]) <= 0.15:
        print('Detected Sentiment Neutral')
    else:
        prediction = np.argmax(score, axis=1)
        if prediction == 0:
            print('Sentiment Detected Negative')
        elif prediction == 1:
            print('Sentiment Detected Positive')
    print('Words Used for Prediction:  {0}'.format(stri))

    data = input('Provide sentence for sentiment Prediction:\n')
    start = time.time()
    predict_sentiment(clean_string=data)
    print('Time taken ', time.time() - start)

def attach_model():
    # load json and create model
    with open('/home/john/sentiment_files/model/movie_pre_trained.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("/home/john/sentiment_files/model/movie_pre_trained.h5")
    print("Loaded model from disk")

    return loaded_model

def load_vocab():
    with open('/home/john/sentiment_files/data/movie_vocab.pkl', 'rb') as f:
        data = pickle.load(f)
    vocab_to_index = data['vocab_to_index']
    index_to_vocab = data['index_to_vocab']
    vocab_frequency = data['vocab_frequency_tuple']
    print('Len of vocab frequency ', len(vocab_frequency))

    return (vocab_to_index, index_to_vocab, vocab_frequency)


if __name__ == '__main__':
    data = input('Provide sentence for sentiment Prediction:\n')
    start = time.time()
    predict_sentiment(clean_string=data)
    print('Time taken {0}'.format(time.time() - start))