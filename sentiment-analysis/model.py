import tensorflow as tf
from tensorflow import keras
# import nlp
import numpy as np
np.random.seed(1337)
import pandas as pd
import warnings
# import nltk
# from nltk.corpus import stopwords

# from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk import word_tokenize
# from nltk.stem.porter import PorterStemmer
# # from nltk.corpus import stopwords, reuters# import re

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
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import metrics, optimizers
from tensorflow.keras.datasets import reuters
import os

companyTweets = pd.read_csv("trainData.csv")

max_length = 100

word_index = None

vocab_size = -1

# cachedStopWords = stopwords.words("english")
# def tokenize(text):
#     min_length = 3
#     words = map(lambda word: word.lower(), word_tokenize(text))
#     words = [word for word in words if word not in cachedStopWords]
#     tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))

#     p = re.compile('[a-zA-Z]+')
#     filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))

#     return filtered_tokens

# def preprocessReuters():
#     # List of document ids
#     documents = reuters.fileids()
    
#     train_docs_id = list(filter(lambda doc: doc.startswith("train"),
#                                 documents))
#     test_docs_id = list(filter(lambda doc: doc.startswith("test"),
#                             documents))
    
#     train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
#     test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]
    
#     # Tokenisation
#     vectorizer = TfidfVectorizer(stop_words=cachedStopWords,
#                                 tokenizer=tokenize)
    
#     # Learn and transform train documents
#     vectorised_train_documents = vectorizer.fit_transform(train_docs)
#     vectorised_test_documents = vectorizer.transform(test_docs)
    
#     # Transform multilabel labels
#     mlb = MultiLabelBinarizer()
#     train_labels = mlb.fit_transform([reuters.categories(doc_id)
#                                     for doc_id in train_docs_id])
#     test_labels = mlb.transform([reuters.categories(doc_id)
#                                 for doc_id in test_docs_id])
#     return (train_docs, train_labels), (test_docs, test_labels)

# def getText(filename):
#     df = pd.read_csv(filename)
#     return df

def preprocessing(text, labels):
    max_words = 5000
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    mat_texts = pad_sequences(sequences, maxlen=255)
    # multilabel_binarizer = LabelEncoder()
    # tags = multilabel_binarizer.fit_transform(labels)
    tags = tf.keras.utils.to_categorical(labels, 3)
    print(tags)
    return mat_texts, tags



def preprocess_prediction(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    return pad_sequences(sequences, maxlen=255)

# def preprocessReutersKeras(text):
#     word_index = reuters.get_word_index()
#     reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
#     decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])   
#     return decoded_newswire
# def vectorize_sequences(sequences, dimension=10000):
#     results = np.zeros((len(sequences), dimension))
#     for i, sequence in enumerate(sequences):
#         results[i, sequence] = 1.

#     return results
def _model():
    model=Sequential()
    model.add(Embedding(5000, 20, input_length=255))
    model.add(Dropout(0.1))
    model.add(Conv1D(64, 3, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPool1D())
    model.add(Dense(256))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
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

# text, labels = nlp.getReuters()
df = pd.read_csv("trainData.csv", encoding="latin1")

headlines = df['Description'].values
sentiments = df['Sentiment'].values
x_train, y_train = preprocessing(headlines, sentiments)
newModel = _model()

newModel.fit(x=x_train, y = y_train, epochs = 3, batch_size=8, validation_split=0.2, shuffle=True)
prediction = ["google got Macaulay Culkin for their home alone inspired ad.. this is the best way to end 2018"]
prediction = preprocess_prediction(prediction)
predictions = newModel.predict(prediction)

print(predictions)

# predictions = preprocess_prediction(np.array(["I think that this is a strong buy for me \n"]))
# print(newModel.predict(predictions))