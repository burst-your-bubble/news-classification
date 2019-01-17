import requests as rq
import json
import time
from sklearn.externals import joblib
import newspaper
from config import api_key
from news_classifier import NewsClassifier

# Get top weekly stories from US News subreddit
def get_topics_r_usnews():
    url = "https://www.reddit.com/r/USNEWS/top.json?t=week"
    res = rq.get(url, headers = {'User-agent': 'Burst Your Bubble'})
    while res.status_code == 429:
        time.sleep(0.1)
        res = rq.get(url, headers = {'User-agent': 'Burst Your Bubble'})
    articles = json.loads(res.text)['data']['children']
    topics = [article['data']['title'] for article in articles]
    return topics

# Get top daily stories from r/POLITICS
def get_topics_r_politics():
    url = "https://www.reddit.com/r/politics/top.json?t=day"
    res = rq.get(url, headers = {'User-agent': 'Burst Your Bubble'})
    while res.status_code == 429:
        print("429")
        time.sleep(0.1)
        res = rq.get(url, headers = {'User-agent': 'Burst Your Bubble'})
    articles = json.loads(res.text)['data']['children']
    topics = [article['data']['title'] for article in articles]
    return topics

# Get topc stories from Politico (via newsapi.org)
def get_topics_politico():
    url = "https://newsapi.org/v2/top-headlines?sources=politico&apiKey={0}"
    url = url.format(api_key)
    res = json.loads(rq.get(url, headers = {'User-agent': 'Burst Your Bubble'}).text)
    topics = [article['title'] for article in res['articles']]
    return topics


# Get articles from newsapi.org by a query string
def get_articles_for_topic(topic):
    url = "https://newsapi.org/v2/everything?q={0}&sortBy=relevance&apiKey={1}"
    url = url.format(topic, api_key)
    res = rq.get(url).text
    data = json.loads(res)
    return data['articles']

# Return article object with classification from newsapi article
def classify_article(article, clf):
    stances = ['L','R','C']

    try:
        a = newspaper.Article(article['url'])
        a.download()
        a.parse()
    except:
        print("Article {0} failed to download".format(article['url']))
        return

    article_text = a.text

    stance = clf.predict(article_text)

    classified = {
        'title': article['title'],
        'author': article['author'],
        'source': article['source']['name'],
        'summary':article['description'],
        'text': article_text,
        'stance': stance,
        'url': article['url'],
        'imageUrl': article['urlToImage']
    }

    return classified

'''
    Return a list of topics and related articles in the following format:

    [
        {
            headline: 
            articles: [
                {
                    title:
                    author:
                    source:
                    summary:
                    text:
                    stance:
                    url:
                    image:url  
                },...
            ]
        },...
    ]
'''
def get_classified_news(clf, src="r/politics"):
    sources = ['r/politics', 'r/usnews', 'politico']
    if src not in sources:
        raise Exception("Invalid source")

    classified_news = []

    if src == "r/politics":
        topics = get_topics_r_politics()
    elif src == "r/usnews":
        topics = get_topics_r_usnews()
    elif src == "politico":
        topics = get_topics_politico()

    i = 1
    for topic in topics:
        print("Topic {0} of {1}: {2}".format(i, len(topics), topic))
        i += 1

        articles = get_articles_for_topic(topic)
        if len(articles) < 6:
            continue

        #classified_articles = [classify_article(a, clf) for a in articles]
        classified_articles = []
        for article in articles:
            ca = classify_article(article, clf)
            if ca is None:
                continue
            classified_articles.append(ca)

        classified_news.append({
            'headline': topic,
            'articles': classified_articles
        })
    
    return classified_news

if __name__ == "__main__":
    model_file_path = "./models/article-classifier_8000x3.pkl"
    clf = NewsClassifier(model_file=model_file_path)

    news = get_classified_news(clf)
    output = json.dumps(news, indent=4, separators=(',',': '))
    print(output)