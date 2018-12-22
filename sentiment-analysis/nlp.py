import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
from nltk.corpus import reuters
import requests

def getReuters():
    return None
    
def getNyTimes():
    r = requests.get("https://api.nytimes.com/svc/search/v2/articlesearch.json?", headers = {"api-key": "7ec31869fad04e73af7e860c82d51042", "q": "Microsoft"})
    dictionary = r.json()
    counter = 0
    print(dictionary)
    

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
    return sent