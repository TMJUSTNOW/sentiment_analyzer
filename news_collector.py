import requests
from bs4 import BeautifulSoup






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

    def yahoo_rss_news(self, stock_symbol_list):
        rss_req_url = 'http://finance.yahoo.com/rss/headline?s={0}'
        resp = requests.get(rss_req_url.format('yhoo'))
        print(resp.content)
        soup = BeautifulSoup(resp.content, 'html.parser')
        print(soup.prettify())


if __name__ == '__main__':
    collect_news().yahoo_rss_news('')