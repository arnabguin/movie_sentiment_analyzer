#!python

import argparse
from lib.constants import model_parms
from numpy import array
from model_loader import ModelLoader
import os
from keras import datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import one_hot

from statistics import mean

class SentimentPredictor:

    word_to_id = datasets.imdb.get_word_index()

    def __init__(self, model):
        self.model = model 
    def predict(self, review):
        encoded_document = [one_hot(review, model_parms.MAX_REVIEW_LENGTH)]
        encoded_padded_document = sequence.pad_sequences(encoded_document, maxlen=model_parms.MAX_REVIEW_LENGTH)
        return self.model.predict(encoded_padded_document)

    def predict_using_imdb_word_index(self, reviews):
        predictions = []
        for review in reviews:
            tmp = []
            for word in review.split(" "):
                tmp.append(self.word_to_id[word])
            tmp_padded = sequence.pad_sequences([tmp], maxlen=model_parms.MAX_REVIEW_LENGTH)
            predictions.append("%s. Sentiment: %s" % (review,model.predict(tmp_padded)))
        return predictions

def predict_from_review(predictor,review):
    prediction = predictor.predict(review)
    print (prediction)
    return prediction


def predict_from_file(predictor,review_file):
    sentiments = []
    with open(review_file, 'r') as rf:
        for r in rf:
            sentiments.append({"review":r, "sentiment":predictor.predict(r)[0][0]})
    print ("review" + "\t" + "sentiment")
    for s in sentiments:
        print (s["review"] + "\t" + str(s["sentiment"]))
    print("Average sentiment score: %.2f" % mean([s['sentiment'] for s in sentiments]))
    return sentiments

def predict_from_review_using_imdb_word_index(predictor):
    bad = "this movie was terrible and bad"
    good = "i really liked the movie and had fun"
    return predictor.predict_using_imdb_word_index([good,bad])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentiment prediction of movie reviews')
    parser.add_argument('--review', '-r', action="store",
                    dest="review", type=str, default=None,
                    help='Review text') 
    parser.add_argument('--review-file', '-rf', action="store",
                    dest="review_file", default=None,
                    help='''Review file containing multiple twitter reviews''')
    options = parser.parse_args()
    if options.review is None and options.review_file is None:
        print("Must provide either review or review file.")
        parser.print_help()
        exit(1)
    model = ModelLoader(model_parms.MODEL_PERSISTENT_NAME).get_loaded_model()
    sentiment_predict = SentimentPredictor(model)
    print ("Doing a simple test prediction ...")
    print(predict_from_review_using_imdb_word_index(sentiment_predict))
    if options.review:
        print ("Predicting sentiment for review " + "'" + options.review + "'")
        predict_from_review(sentiment_predict,options.review)
    if options.review_file:
        print ("Predicting sentiment for review file " + "'" + options.review_file + "'")
        predict_from_file(sentiment_predict,options.review_file)
