import requests
import dateutil.parser as dparser
from datetime import datetime
import multiprocessing
import time
import re
import json
import html
import csv

from keras.models import model_from_json
import nltk
from bs4 import BeautifulSoup
import tweepy
from tweepy.api import API
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import difflib

# from sentiment_api import predict_sentiment

class TweepyListener(tweepy.StreamListener):

    def __init__(self, api=None):
        self.api = api or API()
        self.news_list = []
        df = pd.read_csv('/home/janmejaya/sentiment_analyzer/symbol_to_entity_mapping.csv')
        self.entity_name = [' '.join(val.split()[:3]) for val in df['entityname'].tolist()]


    def on_data(self, encoded_data):
        try:
            decoded_data = json.loads(encoded_data)
            # Remove Hyperlink
            if 'text' in decoded_data:
                tweet = decoded_data['text'].split('https://')[0]
                # Remove words start with @ and hash tags
                tweet = re.sub(' @(.+?) ', ' ', tweet)
                tweet = re.sub(' #(.+?) ', ' ', tweet)
                cleaned_data = re.sub('[^A-Za-z.!? ]+', '', tweet)
                for name in self.entity_name:
                    if name in cleaned_data:
                        self.news_list.append(cleaned_data)
                        print(cleaned_data)
                        print(name)

            if int(time.time()) % 1000 == 0:
                if self.news_list:
                    print('Writting Data to file')
                    with open('live_news_data.txt', mode='a') as file:
                        file.writelines(self.news_list)
                    self.news_list = []
        except Exception as exc:
            print('Exception occurred while processing a tweet. Exception is {0}'.format(exc))

        return True

class collect_news():
    def __init__(self):
        pass

    def newsapi_news_collector(self, company_name_list):
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
        pub_date = []
        company_name = re.sub('[^A-Za-z ]+', '', company_name)
        for each_item in soup.find_all('item'):
            published_date = dparser.parse(str(each_item.pubdate.string))
            if published_date.day == datetime.utcnow().day:
                description = each_item.description.string
                if description:
                    ## Find subject of interest (company which we are intrested)
                    #     # TODO: Find 'NP'-> Noun Phrase from sentence(complex)(it should be company name)
                    #     # Use NLTK for this (http://www.nltk.org/book/ch08.html)
                    taged_description = nltk.tag.pos_tag(nltk.tokenize.word_tokenize(description))
                    for each_tag in taged_description:
                        if each_tag[1] == 'NNP':
                            if each_tag[0] in [company_name.split(' ')[0], stock_symbol]:
                                news_list.append(description)
                                pub_date.append(published_date)
                            break

        return (news_list, pub_date)

    def twitter_time_line_news(self, company_name):
        ## Twitter Data
        api_key = 'BMaRbtElbiTtZiV8B21yD5nAa'
        api_secret_key = 'nYoxVIHJsjhHDpNCESXkKiTOBgrGs4O34QkBtDDAjlshKFaSNs'
        api_access_key = '884591331779031040-qGeQFpCrHGaJFnhCKk4BUIBLn0cWhr1'
        api_access_secret_key = 'QeV2ppw3R71hNvA1zwS5Tmz0t6OXedOF6ma0VAlVLNMUW'

        auth = tweepy.OAuthHandler(api_key, api_secret_key)
        auth.set_access_token(api_access_key, api_access_secret_key)
        api = tweepy.API(auth)

        utc_today_date = datetime.utcnow().day
        company_name = ' '.join(re.sub('[^A-Za-z0-9 ]+', '', company_name).split(' ')[:3])

        news_list = []
        pub_date = []
        news_channel_to_follow = ['CNN', 'businessinsider', 'ft', 'nytimes', 'CNNMoney', 'TwitterBusiness‏', 'FinancialTimes', 'EconBizFin‏', 'ftfinancenews', 'TheEconomist', 'Forbes', 'CNNMoney', 'YahooFinance', 'business', 'WSJ']
        for channel in news_channel_to_follow:
            print(channel)
            try:
                for tweets in tweepy.Cursor(api.user_timeline, screen_name=channel).items():
                    ## gather Today's tweet
                    if tweets.created_at.day != utc_today_date:
                        break
                    news = tweets.text
                    # Remove weblink from news
                    news = news.split('https://')[0]
                    if company_name in news:
                        taged_description = nltk.tag.pos_tag(nltk.tokenize.word_tokenize(news))
                        for each_tag in taged_description:
                            # Filtering 'Simple sentences' which contains provided company name
                            # If first noun phrase encountered doesn't contain company name don't consider that news.
                            if each_tag[1] == 'NNP' and each_tag[0] in company_name:            # Filter out news given stock symbol
                                news_list.append(news)
                                pub_date.append(tweets.created_at)
                                break
            except:
                # print('For {0} len of news list {1}'.format(channel, len(news_list)))
                continue
        print('Final length', len(news_list))
        return (news_list, pub_date)

    def get_stock_data(self, symbol, period, interval, stock_exch='NASD'):
        try:
            query_url = 'https://www.google.com/finance/getprices?q={0}&x={3}&i={2}&f=d,c,v,k,o,h,l&df=cpct&auto=0&ei=Ef6XUYDfCqSTiAKEMg&p={1}'.format(
                                                        symbol, period, interval, stock_exch)
            stock_data_raw = requests.get(query_url).content.decode('utf-8')

            # Formatting Raw stock data
            ## This part is coded as per result returned google query
            start = stock_data_raw.index('COLUMNS=') + len('COLUMNS=')
            end = stock_data_raw.index('\nDATA=')
            column_names = stock_data_raw[start:end]            # Contains Column names separated by comma
            print('Column Names: ', column_names)
            # getting data index
            start_index_str = re.search('TIMEZONE_OFFSET=(.*)\n', stock_data_raw).group(0)
            start = stock_data_raw.index(start_index_str) + len(start_index_str)
            data = stock_data_raw[start:-2]                     # Removing Last '\n'
            data = data.split('\n')
            first_data = data[0].split(',')
            full_time_stamp = int(first_data[0][1:])
            first_data[0] = full_time_stamp

            df = pd.DataFrame(data=[first_data], columns=column_names.split(','))
            for i in range(1, len(data)):
                # convert first element to unix date format
                each_row = data[i].split(',')
                # converting partial time stamp to full time stamp
                each_row[0] = full_time_stamp + int(each_row[0]) * int(interval)
                df = df.append(pd.DataFrame([each_row], columns=column_names.split(',')), ignore_index=True)
            df = df.apply(pd.to_numeric)
            return df
        except Exception as exc:
            print('Exception occurred while getting stock data, EXC: {0}'.format(exc))
            return None

    def collect_live_tweets(self):

        api_key = 'BMaRbtElbiTtZiV8B21yD5nAa'
        api_secret_key = 'nYoxVIHJsjhHDpNCESXkKiTOBgrGs4O34QkBtDDAjlshKFaSNs'
        api_access_key = '884591331779031040-qGeQFpCrHGaJFnhCKk4BUIBLn0cWhr1'
        api_access_secret_key = 'QeV2ppw3R71hNvA1zwS5Tmz0t6OXedOF6ma0VAlVLNMUW'
        title = input('Provide a input to Search separate by comma: ')
        title = [name.rstrip().lstrip() for name in title.split(',')]
        try:
            auth = tweepy.OAuthHandler(api_key, api_secret_key)
            auth.set_access_token(api_access_key, api_access_secret_key)
            overwrite_listener_obj = TweepyListener()
            stream_api = tweepy.Stream(auth=auth, listener=overwrite_listener_obj)
            stream_api.filter(track=title)
        except Exception as exc:
            print('Exception occurred during Authenticating or searching: {0}'.format(exc))

    def collect_historic_tweets(self):
        application_key = 'BMaRbtElbiTtZiV8B21yD5nAa'
        application_secret_key = 'nYoxVIHJsjhHDpNCESXkKiTOBgrGs4O34QkBtDDAjlshKFaSNs'

        # authenticating app
        auth = tweepy.AppAuthHandler(application_key, application_secret_key)
        api = tweepy.API(auth_handler=auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        count = 0
        for tweets in tweepy.Cursor(api.user_timeline, screen_name='YahooFinance').items():
            print(tweets.text)
            print(tweets.created_at)
            count += 1
            print('\nCount ', count)

    def predict_sentiment(self, news_list):
        start = time.time()
        with open('/home/john/sentiment_files/Model_and_data/Model_Aug19/complete_pre_trained.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("/home/john/sentiment_files/Model_and_data/Model_Aug19/complete_pre_trained.h5")
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
                        print('Score: {0}'.format(score[idx][0]))
                    else:
                        print('News:> {0}'.format(each_news))
                        print('Predicted Sentiment: Positive\n')
                        print('Score: {0}'.format(score[idx][1]))
                else:
                    print('News:> {0}'.format(each_news))
                    print('Score {0}'.format(score[idx]))

def collect_data():
    # Get all company name and stock symbol from portfolio
    stock_symbol_df = pd.read_csv('/home/janmejaya/sentiment_analyzer/symbol_to_entity_mapping.csv')
    symbols_list = stock_symbol_df['stock_symbol'].tolist()
    company_names_list = stock_symbol_df['entityname'].tolist()
    news_collector = collect_news()
    today_date = datetime.now().date()
    del stock_symbol_df
    for idx, company_name in enumerate(company_names_list):
        print('For Company Name ', company_name)
        twitter_news_list, twitter_pub_date = news_collector.twitter_time_line_news(company_name=company_name)
        rss_news_list, rss_pub_date = news_collector.yahoo_rss_news(stock_symbol=symbols_list[idx])

        # TODO: get information regarding time zone of twitter and rss data, convert it to NY time zone as we are using NASDAQ exchange's stock data
        if twitter_pub_date or rss_pub_date:
            min_date = min(twitter_pub_date + rss_pub_date)
            delta_date = today_date - min_date
            period = delta_date.days
            df = []
            if period:
                df = news_collector.get_stock_data(symbol=symbols_list[idx], period=period, interval='3600')

            if df:
                # TODO: get if date is with in two hour time period get that news and stock price
                



def _test_yahoo_rss_news():
    stock_symbol = input('Please Provide a Stock Symbol: ')
    news_collector = collect_news()
    news_list, _ = news_collector.yahoo_rss_news(stock_symbol=stock_symbol)
    if news_list:
        news_collector.predict_sentiment(news_list)
    else:
        print('No news Found for Stock Symbol: {0}'.format(stock_symbol))

def _test_twitter_timeline_news():
    stock_symbol = input('Please Provide a Stock Symbol: ')
    symbol_lookup_url = 'http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en'
    symbol_resp = requests.get(symbol_lookup_url.format(stock_symbol)).json()
    company_name = ''
    for result_dict in symbol_resp['ResultSet']['Result']:
        if result_dict['exchDisp'] in ['NASDAQ', 'NYSE']:
            company_name = result_dict['name']
            break

    if company_name:
        news_collector = collect_news()
        news_list, _ = news_collector.twitter_time_line_news(company_name=company_name)
        if news_list:
            news_collector.predict_sentiment(news_list)
        else:
            print('No news Found for Stock Symbol: {0}'.format(stock_symbol))
    else:
        print('Unable to find any company name related to given Stock Symbol: {0}'.format(stock_symbol))
    

if __name__ == '__main__':
    # _test_yahoo_rss_news()
    # collect_news().twitter_news_collector(stock_symbol)
    # collect_news().collect_live_tweets()
    # collect_news().collect_historic_tweets()
    # collect_news().get_stock_data('AAPL', '2d', '3600')
    collect_data()