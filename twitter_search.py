import tweepy as tp
import pprint
import argparse
import json
import nltk
import ssl
import re
from nltk.stem.porter import PorterStemmer
from lib.constants import model_predicates
from textblob import TextBlob
from tabulate import tabulate


nltk.download("stopwords")
from nltk.corpus import stopwords

from collections import namedtuple

pp = pprint.PrettyPrinter(indent=4)

class TwitterSearchEngine:
    def __init__(self, query, query_limit, creds_file):
        self.query = query + " -filter:retweets"
        self.query_limit = query_limit
        with open(creds_file) as c_file:
            self.credentials = json.load(c_file)
        self.auth()

    def auth(self):
        model_predicates.authorize()
        # Authenticate to Twitter
        auth = tp.auth.OAuthHandler(self.credentials.get('api_key'), self.credentials.get('api_secret'))
        auth.set_access_token(self.credentials.get('access_token'), self.credentials.get('access_token_secret'))

        self.api = tp.API(auth, wait_on_rate_limit=True)

        try:
            self.api.verify_credentials()
            #print("Authentication verified OK. Good to go.")
        except:
            print("Error during authentication. Please correct security credentials and try again")

    def clean_text(self, text, remove_stopwords=False):
    
        text = text.lower().split()

        if remove_stopwords:
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
    
        text = " ".join(text)

        text = re.sub(r"<br />", " ", text)
        text = re.sub(r"   ", " ", text) # Remove any extra spaces
        text = re.sub(r"  ", " ", text)
        stemmer = PorterStemmer()
        text = [stemmer.stem(word) for word in text]
        text = "".join(text)

        ### text = re.sub(r"[^a-z]", " ", text)  ## do not remove non-alphabet characters other some sentiment
        ###                                      ## strings get removed as well

        return(text)

    def get_results(self):
        data = {}
        data['tweets'] = []
        for tweet in tp.Cursor(self.api.search_tweets, q=self.query, lang='en').items(self.query_limit):
            cleaned_text = self.clean_text(tweet.text, True)
            blob = TextBlob(cleaned_text)
            sentiment = blob.sentiment
            if sentiment.subjectivity >= 0.75:
                data['tweets'].append({
                    'tweet' : cleaned_text,
                    'polarity' : sentiment.polarity,
                    'subjectivity' : sentiment.subjectivity
                })
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Twitter search of movie reviews')

    parser.add_argument('--credentials-file', '-c', action="store",
                    dest="auth_credentials_file", default=None, 
                    help='''Json file storing auth credentials.
                    Make sure you create a JSON file in the following format and fill in the right keys.
                    eg. security.json
                    {
                         'api_key' : <YOUR_API_KEY>,
                         'api_secret' : <YOUR_API_SECRET>,
                         'access_token' : <YOUR_ACCESS_TOKEN>,
                         'access_token_secret' : <YOUR_ACCESS_TOKEN_SECRET>
                    }
                    ''')
    parser.add_argument('--query', '-q', action="store",
                    dest="query", default=None,
                    help='''Query string. The user may provide the following types of queries:
                            -*- Name of movie only (eg. theking) 
                            -*- Name of movie plus augmentation keywords (eg. charlies angels movie critic) 
                            -*- Twitter hashtag (eg. #missionimpossible2)''')
    parser.add_argument('--query-limit', '-ql', action="store",
                    dest="query_limit", type=int, default=1000,
                    help='Count limit on number of lines in query result')

    parser.add_argument('--remove-stop-words', '-rs', default=False, action="store_true",
                    dest="remove_stop_words", 
                    help='Remove nltk stop words from reviews')

    parser.add_argument('--no-score', '-nr', default=False, action="store_true",
                    dest="no_score", 
                    help='Print only tweets (without score)')

    options = parser.parse_args()

    if options.auth_credentials_file is None:
        print ("Missing credentials file.")
        parser.print_help()
        exit(1)
    if options.query is None:
        print ("Must specify a query")
        parser.print_help()
        exit(1)

    query = options.query
    query_limit = options.query_limit
    creds_file = options.auth_credentials_file

    tsearch = TwitterSearchEngine(query, query_limit, creds_file)
    #print ("==========================")
    tweets = tsearch.get_results()
    if options.no_score:
        print ("\n".join([t['tweet'] for t in tweets['tweets']]))
    else:
        pp.pprint (tsearch.get_results())
