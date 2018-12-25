import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from agent.agent import Agent
from functions import *
import sys

if len(sys.argv) != 3:
	print("Usage: python predict.py [stock] [model]")
	exit()

stock_name, model_name = sys.argv[1], sys.argv[2]

data = getStockDataPdr(stock_name, 253)
model = load_model("models/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]
state = getState(data, 252, window_size+1) 

action = agent.act(state)

# sit
next_state = getState(data, 253, window_size + 1)
reward = 0

if action == 1: # buy
    agent.inventory.append(data[t])
    print("Buy: " + formatPrice(data[t]))

elif action == 2 and len(agent.inventory) > 0: # sell
    bought_price = agent.inventory.pop(0)
    reward = max(data[t] - bought_price, 0)
    total_profit += data[t] - bought_price
    print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

else:
    print("Hold the stock today")


