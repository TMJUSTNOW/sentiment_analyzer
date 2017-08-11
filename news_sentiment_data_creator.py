import requests

from bs4 import BeautifulSoup



def hindu_news_crawler():
    url ='http://www.thehindu.com/search/?order=DESC&page=1&q=apple+inc&sort=publishdate'

    resp = requests.get(url)
    soup = BeautifulSoup(resp.content, 'html.parser')
    # print(soup.prettify())
    all_class = soup.find_all('div', class_='feature-news')
    for i in all_class:
        print(i.get('main'))









if __name__ == '__main__':
    hindu_news_crawler()