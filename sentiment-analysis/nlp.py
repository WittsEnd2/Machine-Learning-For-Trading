import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
from nltk.corpus import reuters
import time
# from tensorflow.keras.datasets import reuters
import requests
import threading

    
def getNyTimes(query, page):
    returnArr = []
    payload = {"q": query, "fl": "headline", "page": str(page)}
    
    r = requests.get("https://api.nytimes.com/svc/search/v2/articlesearch.json?", headers = {"api-key": "7ec31869fad04e73af7e860c82d51042"}, params = payload)
    dictionary = r.json()
    return dictionary
    
def getNyTimesSentiment(nytimes):
    print(len(nytimes['response']['docs']))
    f = open("headlines.csv", "a")
    for i in nytimes["response"]["docs"]:
        headline = str(i['headline']['main'])
        headline = headline.strip("\n")
        headline = headline.strip("\.")
        headline = headline.replace("\n",". ")
        headline = headline.replace("\\'","'")
        headline = headline.replace("\\","")
        headline = headline.replace("\\\.",".")
        headline = headline.replace("\"", "'")
        headline = headline.replace("\\n",". ")
        headline = headline.replace ("\\,", "")
        headline = headline.replace(",", "")
        f.write(headline + "," + getSentiment(i['headline']['main']) + "\n")
    f.close()

def getReuters():

    documents = reuters.fileids()
    train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
    test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))
    train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
    print(len(train_docs))
    print(len(train_docs[0]))
    test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]
    labels = []
    for i in train_docs:
        labels.append(getSentiment(i))
    return train_docs, labels
    
def getSentiment(tweet):
    sid = SentimentIntensityAnalyzer()
    sent = 0.0
    count = 0
    sentList = nltk.tokenize.sent_tokenize(tweet)

    # Go through each sentence in tweet
    for sentence in sentList:
        count += 1
        ss = sid.polarity_scores(sentence)
        sent += ss['compound']  # Tally up the overall sentiment

    if count != 0:
        sent = float(sent / count)
    if (sent > 0):
        return "Buy"
    elif (sent < 0):
        return "Sell"
    else:
        return "Hold"

def execute(company): 
    for i in range(0, 20):
        time.sleep(1)
        getNyTimesSentiment(getNyTimes(company, i))


for i in range(0, 20):
    time.sleep(1)
    getNyTimesSentiment(getNyTimes("Microsoft", i))
for i in range(0, 20):
    time.sleep(1)
    getNyTimesSentiment(getNyTimes("Google", i))
for i in range(0, 20):
    time.sleep(1)
    getNyTimesSentiment(getNyTimes("Nvidia", i))
for i in range(0, 20):
    time.sleep(1)
    getNyTimesSentiment(getNyTimes("Intel", i))
for i in range(0, 20):
    time.sleep(1)
    getNyTimesSentiment(getNyTimes("Apple", i))
for i in range(0, 20):
    time.sleep(1)
    getNyTimesSentiment(getNyTimes("Facebook", i))
