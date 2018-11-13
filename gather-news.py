import requests as rq
import json
import time
from sklearn.externals import joblib
import newspaper

model_file_path = "./models/article-classifier.pkl"

# Get top weekly stories from US News subreddit
def get_reddit_data():
    url = "https://www.reddit.com/r/USNEWS/top.json?t=week"
    res = rq.get(url)
    while res.status_code == 429:
        time.sleep(0.1)
        res = rq.get(url)
    data = json.loads(res.text)
    return data

# Get the top US news articles from newsapi.org
#   PROMBLEM: Can't filter by 'politics' category, only 'general'
def get_newsapi_top_articles():
    url = "https://newsapi.org/v2/top-headlines?country=us&category=general&apiKey={0}"
    api_key = "390c49179a5f475387ea9286be276473"
    url = url.format(api_key)
    res = rq.get(url).text
    data = json.loads(res)
    articles = [{'title':a['title'], 'url':a['url']} for a in data['articles']]
    return articles


# Get articles from newsapi.org by a query string
def hit_news_api(query):
    url = "https://newsapi.org/v2/everything?q={0}&sortBy=relevance&apiKey={1}"
    api_key = "390c49179a5f475387ea9286be276473"
    url = url.format(query, api_key)
    res = rq.get(url).text
    data = json.loads(res)
    return data['articles']

def get_classified_news_articles():
    topics = []
    classified_articles = []

    data = get_reddit_data()['data']
    posts = data['children']   
    post_titles = [post['data']['title'] for post in posts]
    for t in post_titles:
        articles = hit_news_api(t)
        articles = [{'title':a['title'], 'url':a['url']} for a in articles]
        topics.append({"headline":t, 'articles':articles})

    clf = joblib.load(model_file_path)
    stances = ['L','R','C']
    
    for topic in topics:
        articles = []
        for article in topic['articles']:
            try:
                article_text = newspaper.Article(article['url'])
                article_text.download()
                article_text.parse()
            except:
                continue
            text = article_text.text
            text = text.replace("\n", "")
            stance = clf.predict([text])[0]
            classified_article = {'url':article['url'], 'title':article['title'], 'stance':stances[stance]}
            articles.append(classified_article)
        classified_articles.append({'headline':topic['headline'], 'articles':articles})
        
    
    return classified_articles

if __name__ == "__main__":
    topics = get_classified_news_articles()
    print(json.dumps(topics, indent=4, separators=(',',': ')))