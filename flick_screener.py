#!python
 
import argparse
import os
from lib.constants import model_parms, model_predicates
from statistics import mean


def analyze_tweet(tweet):
    import sentiment_predictor
    from model_loader import ModelLoader
    model = ModelLoader(model_parms.MODEL_PERSISTENT_NAME).get_loaded_model()
    predictor = sentiment_predictor.SentimentPredictor(model)
    sentiment_predictor.predict_from_review(predictor, tweet)


def analyze_movie(movie_name):
    import get_movie
    import sentiment_predictor
    import twitter_search
    from model_loader import ModelLoader
    import re

    movie_details = get_movie.get_movie(movie_name)
    print(movie_details)
    print("===================================================")
    #print("MetaScore {}]\tImdb Rating {}".format(movie_details['metascore'], movie_details['imdb_rating']))
    tweets = twitter_search.TwitterSearchEngine(movie_name, 100, "security.json").get_results()
    normalized_title = re.sub(r'[^a-zA-Z0-9]', '', movie_name);
    tweets_file = "{}.txt".format(normalized_title)
    with open(tweets_file, "w") as tf:
        tf.write("\n".join([t['tweet'] for t in tweets['tweets']]))
    model = ModelLoader(model_parms.MODEL_PERSISTENT_NAME).get_loaded_model()
    predictor = sentiment_predictor.SentimentPredictor(model)
    sentiment_predictor.predict_from_file(predictor, os.getcwd() + "/" + normalized_title + ".txt")


if __name__ == '__main__':
    model_predicates.authorize()
    parser = argparse.ArgumentParser(description='Sentiment prediction of movie reviews from tweets')
    parser.add_argument('--movie', '-m', action="store",
                        dest="movie_name", type=str, default=None,
                        help='Full or Partial name of the movie')
    parser.add_argument('--tweet', '-t', action="store",
                        dest="tweet", type=str, default=None,
                        help='Some Tweet or Text for predicting sentiment')
    options = parser.parse_args()
    if options.movie_name is None and options.tweet is None:
        print("Please use right syntax:\n \tpython flick_screener.py -m \"<movie name>\" \n OR \n \t python "
              "flick_screener.py -t \"<some_tweet> \"")
        parser.print_help()
        exit(1)
    if options.movie_name is not None:
        analyze_movie(options.movie_name)
    else:
        analyze_tweet(options.tweet)
