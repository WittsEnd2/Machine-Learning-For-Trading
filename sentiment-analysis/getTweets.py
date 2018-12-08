from tweepy import Stream
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
import tweepy
import emoji
import unicodedata
import json
import datetime
import nlp

'''
- What to do next
1) Create basic NLTK algorithm to have a base model for training data
2) Gather tweets and have it go through the NLTK algorithm
3) After a certain amount of texts classified, run it through ML algorithm
4) Combine both the ML algorithm from price prediction and news forecasting to create a refined tool to buy/sell stocks
5) Finish
'''

consumer_key = "kEaMOwaPFjdwXelB8rwMXh1Yg"
consumer_secret = "2AKsg0BwxI1bDErlQDUgmNVDvOwzx98htaiYKGXeZzoHv71Jy6"

access_token = "839692039-VulkCGE4QZRZlKYQtDuNjRWgJnxsCdeSfqaPuObs"
access_token_secret = "08AltS6hMTj5Y7sDh2cSLZCrtuqfEYQJ5LTqEI2N4FLN0"
company = "Microsoft"
ticker_symbol = "JetBlue"
companyTweets = open("companyData.csv", "a")

totalSentiment = 0

class listener(StreamListener):
	def __init__(self):
		super().__init__()
		self.counter = 0 
	# def on_status(self, status):
	# 	print(status)
	# def on_data(self, data):
	# 	print(data)
	def on_data(self, data):
		
		data = str(emoji.demojize(data))
		
		decoded = json.loads(str(data))
		# if 'place' in decoded and decoded['place'] is not None:
			# loc = decoded['place']['bounding_box']['coordinates'][0][0]
			
		tweet = str(emoji.demojize(decoded['text']).encode("unicode_escape"))
		tweet = tweet[1:]
		tweet = tweet.strip("\n")
		tweet = tweet.strip("\.")

		tweet = tweet.replace("\n",". ")
		tweet = tweet.replace("\\'","'")
		tweet = tweet.replace("\\","")
		tweet = tweet.replace("\\\.",".")
		tweet = tweet.replace("\"", "'")
		tweet = tweet.replace("\\n",". ")
		tweet = tweet.replace ("\\,", "")
		tweet = tweet.replace(",", "")
		print(tweet)
		sentimentNum = nlp.getSentiment(tweet)
		sentiment = 0
		if (sentimentNum > 0.8):
			sentiment = 1
		elif (sentimentNum < -0.8):
			sentiment = -1
		else:
			sentiment = 0
		companyTweets.write(str(datetime.datetime.now()) + ', ' + tweet + ', ' + str(sentimentNum) + '\n')
		companyTweets.flush()
		tweetLower = tweet.lower()
	def on_error(self, status):
		print(status)

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

myStreamListener = listener()
myStream = Stream(auth = api.auth, listener=myStreamListener)
myStream.filter(languages=["en"], track=["Microsoft", "Google", "Facebook", "Dow Jones", "Trade War"])

