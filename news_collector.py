import requests
from bs4 import BeautifulSoup
import dateutil.parser as dparser
from datetime import datetime
from keras.models import model_from_json
import time
import re
import nltk

from sentiment_api import predict_sentiment

import difflib




class collect_news():
    def __init__(self):
        pass

    def gather_news(self, company_name_list):
        api_key = '58f4449443e248acb31ff8bdfe0e48f7'
        source = 'bloomberg'
        newsapi_url = 'https://newsapi.org/v1/articles?source={0}&sortBy=top&apiKey={1}'
        json_response = requests.get(newsapi_url.format(source, api_key)).json()
        print(json_response)

    def news_web_query(self):
        URL = 'https://www.google.com/search?pz=1&cf=all&ned=us&hl=en&tbm=nws&gl=us&as_q={0}&as_occt=any&as_drrb=b&as_mindate={3}%2F%{1}%2F{4}&as_maxdate={3}%2F{2}%2F{4}&tbs=cdr%3A1%2Ccd_min%3A3%2F1%2F13%2Ccd_max%3A3%2F2%2F13&as_nsrc=Gulf%20Times&authuser=0'
        response = requests.get(URL.format('Apple Inc', 8, 9, 8, 17))
        print('URL:\n{0}'.format(URL.format('Apple Inc', 8, 9, 8, 17)))
        print(response.content)

    def yahoo_rss_news(self, stock_symbol='aapl'):
        rss_req_url = 'http://finance.yahoo.com/rss/headline?s={0}'
        resp = requests.get(rss_req_url.format(stock_symbol))
        soup = BeautifulSoup(resp.content, 'html.parser')

        symbol_lookup_url = 'http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en'
        symbol_resp = requests.get(symbol_lookup_url.format(stock_symbol)).json()
        company_name = ''
        for result_dict in symbol_resp['ResultSet']['Result']:
            if result_dict['exchDisp'] in ['NASDAQ', 'NYSE']:
                company_name = result_dict['name']
                break
        news_list = []
        company_name = re.sub('[^A-Za-z ]+', '', company_name)
        for each_item in soup.find_all('item'):
            if dparser.parse(str(each_item.pubdate.string)).day == datetime.utcnow().day:
                description = each_item.description.string
                if description:
                    ## Find subject of interest (company which we are intrested)
                    #     # TODO: Find 'NP'-> Noun Phrase from sentence(simple/complex)(it should be company name)
                    #     # Use NLTK for this (http://www.nltk.org/book/ch08.html)
                    taged_description = nltk.tag.pos_tag(nltk.tokenize.word_tokenize(description))
                    for each_tag in taged_description:
                        if each_tag[1] == 'NNP':
                            if each_tag[0] in [company_name.split(' ')[0], stock_symbol]:
                                news_list.append(description)
                            break

        start = time.time()
        with open('/home/janmejaya/sentiment_files/Model_and_data/complete_sentiment_15_word_new.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("/home/janmejaya/sentiment_files/Model_and_data/complete_sentiment_15_word_new.h5")
        print("Loaded model from disk")
        print("Time Taken to load model: {0}".format(time.time() - start))
        start2 = time.time()
        if news_list:
            score = predict_sentiment(model=loaded_model, clean_string_list=news_list)
            print('Time for sentiment prediction: {0}'.format(time.time() - start2))
            for idx, each_news in enumerate(news_list):

                if abs(score[idx][0] - score[idx][1]) >= 0.15:
                    if score[idx][0] > score[idx][1]:
                        print('News:> {0}'.format(each_news))
                        print('Predicted Sentiment: Negative\n')
                    else:
                        print('News:> {0}'.format(each_news))
                        print('Predicted Sentiment: Positive\n')
        else:
            print('No news Found for Stock Symbol: {0}'.format(stock_symbol))

if __name__ == '__main__':
    stock_symbol = input('Please Provide a Stock Symbol: ')
    collect_news().yahoo_rss_news(stock_symbol=stock_symbol)