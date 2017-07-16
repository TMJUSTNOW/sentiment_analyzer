import os
import json
import numpy as np
import html.parser
import re



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

    # Decode to utf8
    decoded_data = html_cleaned_data.decode("utf8")

    # Keep important punctuation
    unimportant_punctuation =['*', '^', '*', '+', '|', '%', '@', '$', '~', '`']
    string_for_sub = '|'.join(unimportant_punctuation)
    remove_punctuation = re.sub(string_for_sub, '', decoded_data)

    # Break data into words

    # Convert data into standard lexicons
