import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
from nltk.corpus import reuters



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

def trainOnReuters(): 
    train_feats = []
    test_feats = []
	
    for fileid in reuters.fileids():
        if fileid.startswith('training'):
            featlist = train_feats
        else: # fileid.startswith('test')
            featlist = test_feats
		
        feats = feature_detector(reuters.words(fileid))
        labels = reuters.categories(fileid)
        featlist.append((feats, labels))
	
    return train_feats, test_feats 