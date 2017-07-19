import pickle

from keras.models import model_from_json
from nltk.tokenize import word_tokenize
import numpy as np
import tensorflow as tf







def predict_sentiment(clean_string):

    vocab_to_index, index_to_vocab, vocab_frequency = load_vocab()
    imdb_model = attach_model()
    # Model Parameters
    max_word = 50
    max_features = 20000
    # evaluate loaded model on test data
    imdb_model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    frequent_words_tuple = sorted(vocab_frequency.items(), key=lambda x: x[1], reversed=True)[:max_features]
    frequent_words = [val for val, freq in frequent_words_tuple]
    tokenized_word = word_tokenize(clean_string)
    input_data = np.zeros([1, max_word], dtype=np.int32)
    truncated_data = []
    for each_word in tokenized_word:
        if each_word in frequent_words:
            truncated_data.append(vocab_to_index[each_word])
    # Perform Extra Padding
    if len(truncated_data) < max_word:
        truncated_data += [0] * (max_word - len(truncated_data))

    input_data[0] = truncated_data
    prediction = imdb_model.predict(input_data)
    sess = tf.Session()
    prediction = np.argmax(sess.run(tf.nn.softmax(prediction)), axis=1)
    stri = ''
    for idx in input_data[0]:
        stri += ' ' + index_to_vocab[idx]
    if prediction == 0:
        print('Sentiment Detected Negetive')
    elif prediction == 1:
        print('Sentiment Detected Positive')
    print('Words Used for Prediction:  {0}'.format(stri))

def attach_model():
    # load json and create model
    with open('model/imdb_lstm.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model/imdb_lstm.h5")
    print("Loaded model from disk")

    return loaded_model

def load_vocab():
    with open('data/movie_vocab.pkl', 'rb') as f:
        data = pickle.load(f)
    vocab_to_index = data['vocab_to_index']
    index_to_vocab = data['index_to_vocab']
    vocab_frequency = data['vocab_frequency']

    return (vocab_to_index, index_to_vocab, vocab_frequency)


if __name__ == '__main__':
    data = input('Provide sentence for sentiment Prediction:\n')
    predict_sentiment(clean_string=data)