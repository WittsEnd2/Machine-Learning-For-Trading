import numpy as np
import pandas_datareader as pdr
import pandas_datareader.data as web
from datetime import datetime, timedelta
import math

# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	vec = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(float(line.split(",")[4]))
	return vec

def getStockDataPdr(stockTicker, hm_days): 
	vec = []
	start = datetime.now() - timedelta(days=hm_days)
	end = datetime.now()
	f = web.DataReader(stockTicker, 'iex', start, end)
	
	for index, value in f.iterrows():
		vec.append(value.close)
	return vec
	
# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res])