from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

import pandas as pd
import nltk, re, time
from nltk.corpus import stopwords
from string import punctuation
import os
import tensorflow as tf

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from lib.constants import model_parms

nltk.download('stopwords')
max_features = model_parms.MAX_NUM_WORDS
# cut texts after this number of words (among top max_features most common words)
batch_size = model_parms.MODEL_TRAIN_BATCH_SIZE 

# Load the data
train = pd.read_csv("labeled_data", delimiter="\t")

# Clean and Format the Data
def clean_text(text, remove_stopwords=True):
    '''Clean the text, with the option to remove stopwords'''
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"[^a-z]", " ", text)
    text = re.sub(r"   ", " ", text) # Remove any extra spaces
    text = re.sub(r"  ", " ", text)
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Return a list of words
    return(text)


# Clean the training and testing reviews
train_clean = []
for review in train.review:
    train_clean.append(clean_text(review))

# Tokenize the reviews
all_reviews = train_clean 
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_reviews)
print("Fitting is complete.")

train_seq = tokenizer.texts_to_sequences(train_clean)
print("train_seq is complete.")

# Find the number of unique tokens
word_index = tokenizer.word_index
print("Words in index: %d" % len(word_index))

# Pad and truncate the reviews so that they all have the same length.
max_review_length = model_parms.MAX_REVIEW_LENGTH

train_pad = sequence.pad_sequences(train_seq, maxlen = max_review_length)
print("train_pad is complete.")

# Create the training and validation sets
x_train, x_test, y_train, y_test = train_test_split(train_pad, train.sentiment, test_size = 0.15, random_state = 2)
print('length training set',len(x_train))
print('length training set sentiments',len(y_train))

print('Loading data...')
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, model_parms.MAX_EMBEDDING_OUTPUT_DIM_LENGTH))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.metrics_names)
print('Start training ...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          validation_data=(x_test, y_test))
print('End training ...')
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

model.save(model_parms.MODEL_PATH)
print("Successfully saved model %s" % model_parms.MODEL_PATH)

neg = 0
pos = 0
counter = 0
word_to_id = imdb.get_word_index()

rounded_predictions = model.predict(x_test,batch_size=batch_size)
counter = len(rounded_predictions)

num_positives = sum(rounded_predictions)
print(f'Number of positive predictions (1s): {num_positives}/{counter}')
print(f'Number of negative predictions (0s): {len(rounded_predictions) - num_positives}/{counter}')

print ("Sample prediction using IMDB word index ...")
bad = "this movie was terrible and bad"
good = "i really liked the movie and had fun"
for review in [good,bad]:
    tmp = []
    for word in review.split(" "):
        tmp.append(word_to_id[word])
    tmp_padded = sequence.pad_sequences([tmp], maxlen=model_parms.MAX_REVIEW_LENGTH)
    print("%s. Sentiment: %s" % (review,model.predict(tmp_padded)))
