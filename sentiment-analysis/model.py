import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

# import nltk
# from nltk.corpus import stopwords

# from nltk.corpus import stopwords
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D, GlobalMaxPool1D
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import metrics, optimizers
import os

companyTweets = pd.read_csv("trainData.csv")

max_length = 100

word_index = None

vocab_size = -1

def getText(filename):
    df = pd.read_csv(filename)
    return df

def preprocessing(df):
    tokenizer = Tokenizer(num_words=5000)
    description = df['Description'].values
    sentiment = df['Sentiment'].values
    print(sentiment)
    tokenizer.fit_on_texts(description)
    sequences = tokenizer.texts_to_sequences(description)
    mat_texts = pad_sequences(sequences, maxlen=255)

    multilabel_binarizer = LabelBinarizer()
    tags = multilabel_binarizer.fit_transform(sentiment)
    print(tags)
    return mat_texts, tags

def preprocess_prediction(text):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    return pad_sequences(sequences, maxlen=255)
    
def _model():
    model=Sequential()
    model.add(Embedding(5000, 20, input_length=255))
    model.add(Dropout(0.1))
    model.add(Conv1D(300, 3, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPool1D())
    model.add(Dense(3))
    model.add(Activation('sigmoid'))
    model.summary()
    adam = optimizers.Adam()
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])

    return model

def shuffle(df):
    return df.reindex(np.random.permutation(df.index))
df = getText("trainData.csv")
df = shuffle(df)
x_train, y_train = preprocessing(df)
newModel = _model()
newModel.fit(x=x_train, y = y_train, epochs = 10, batch_size=32, validation_split=0.2, shuffle=True)

predictions = preprocess_prediction(np.array(["THIS IS AMAZING AND SOOOO GOOD!"]))
print(newModel.predict(predictions))