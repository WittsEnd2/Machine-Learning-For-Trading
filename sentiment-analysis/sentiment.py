import firebase_admin
from firebase_admin import credentials
import csv, random
import nltk
import tweet_features, tweet_pca
import time
from urllib.parse import quote
import json

# read all tweets and labels
fp = open( 'TrainingSet.txt', 'rt' )
reader = csv.reader( fp, delimiter='\t', quotechar='"', escapechar='\\' )
tweets = []
print(reader)
for row in reader:
    tweets.append( [row[1], row[0]] )
    print(row[1])
print(tweets[0:50])

# treat neutral and irrelevant the same
for t in tweets:
    print(t[0])
    if t[0] == 'irrelevant':
        t[0] = 'neutral'


# split in to training and test sets
random.shuffle( tweets )

fvecs = [(tweet_features.make_tweet_dict(s),t) for (t,s) in tweets]
v_train = fvecs[:375]
v_test  = fvecs[375:]
#print(str(v_train))
#print(str(v_test))
for i in range(0, 20):
    print(fvecs[i])

# dump tweets which our feature selector found nothing
tot = 0
for i in range(0,len(tweets)):
    if tweet_features.is_zero_dict( fvecs[i][0] ):
        #print(tweets[i][1] + ': ' + tweets[i][0])
        tot = tot + 1
print(tot)

# apply PCA reduction
#(v_train, v_test) = \
#        tweet_pca.tweet_pca_reduce( v_train, v_test, output_dim=1.0 )


# train classifier
classifier = nltk.NaiveBayesClassifier.train(v_train);
#classifier = nltk.classify.maxent.train_maxent_classifier_with_gis(v_train);


# classify and dump results for interpretation
print("\nAccuracy " + str(nltk.classify.accuracy(classifier, v_test)) +"\n" )#% nltk.classify.accuracy(classifier, v_test)
#print classifier.show_most_informative_features(200)


# build confusion matrix over test set
test_truth   = [s for (t,s) in v_test]
test_predict = [classifier.classify(t) for (t,s) in v_test]

companyTweets = open("companyTweets.csv", "r")


db = firebase.FirebaseApplication("https://glowing-inferno-6337.firebaseio.com/Tweets")

while 1:
    whereCompanyTweet = companyTweets.tell()

    lineCompanyTweet = companyTweets.readline()

    if lineCompanyTweet != None and len(lineCompanyTweet) != 0:
        print(lineCruz)
        lineCompanyTweet = str(lineCompanyTweet)
        decoded = lineCompanyTweet.split(",")
        data = {}
        companyTweetTest = tweet_features.make_tweet_dict(decoded[1])
        data["attitude"] = str(classifier.classify(companyTweetTest))
        data["date"] = data[0]
        data["tweet"] = decoded[1]
        print("Hei, man please I tried")
        db.post('/CompanyTweet', data)
        #db.put('/Tweets', quote(str(data)), None)  
    else:
        time.sleep(1)
        companyTweets.seek(whereCompanyTweet)
#print('Confusion Matrix')
#print (str(nltk.ConfusionMatrix( test_truth, test_predict )))
