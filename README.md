
MS UIUC, Fall 2019 ML project

Arnab Guin
Nishant Garg
Srinivasan Manoharan

### Project Goal

Perform sentiment Analysis on tweets for any movie , and compares the result with IMDB Ratings and Metacritic Ratings.

### Introduction

The objective of the project is to predict movie review sentiments. For a movie
selected by an user, we first fetch a review score of the movie from IMDB and Metacritic
and then compare it with the review score output by our own sentiment
analyzer that scans Twitter tweets on the movie and runs prediction on a trained
classifier model based also on Twitter tweets on a known movie names dataset.

### Model Training

We initially experimented with IMDB movie ratings for our model training and ran
into some accuracy issues since the IMDB data set did not match the Twitter
data in the nature, length and type of reviews.

So we used a pre-determined set of movies, scanned Twitter for tweets and generated
training data based on Twitter's view of sentiment scores. Following were the steps 
followed to generated the labeled dataset:
1. Create a list of pre-determined movies (The movies were selected from IMDB
top 100 and worst 100 list of movies) and store them in a file called
movies.list
2. Use generate_labeled_data.py to generate labeled_data
3. generate_labeled_data.py uses twitter_search.py to first fetch all the 
tweets about each movie. twitter_search.py filters the tweets based on subjectivity
where we remove all scores which have subjectivity of 0. generate_labeled_data.py then
further filters the tweets and considers tweets that have the polarity of over
0.25. This allows us to reduce a lot of noise from the tweets.
4. This process can be improved further, but considering only those tweets that have
a higher subjectivity/polarity, but we did not have an opportunity to try that
(creating labeled data set takes a lot of time, especially because one keeps getting
rate-limited from twitter every few mins)
5. The labeled_data is subsequently consumed by the model for training/testing

Once the training data was cleaned and ready for use, it was fed to a "neural network"
model for training

### Sentiment Prediction

Given a new movie, we search for tweets on that movie and pass them through a
sentiment analyzer tool that uses our pre-trained model for sentiment prediction.

### Software Packages

Python 3.8.0
Tweepy
Nltk
Ssl
Keras
Urllib
Numpy
Textblob

### Environment Setup

pip install -r requirements.txt

### Training Dataset Generation


### How To Run
- `Step 1: Mandatory` Get twitter developer account/oauth2 credentials 
  - visit https://developer.twitter.com/
  - the credentials have to be stored in a file called `security.json` in the following format
    ```
    { 
        "api_key" : "<val>", 
        "api_secret" : "<val>", 
        "access_token": "<val>", 
        "access_token_secret": "<val>"
    }
    ```

- `Step 2: Optional` Build a labeled dataset  

  `python3 generate_labeled_data.py`
  
  The above code will read the list if files from movies.list(100 tweets per movie). If 
  more tweets are needed to create a larger dataset, then that has to be done manually

  The labeled data is in the following tab separated format:
  <review>	<sentiment>

  FYI, the labeled_data that has been checked in, has been created with 1000 tweets/movie

- `Step 3: Optional` Generate a model with the above mentioned dataset
  We have tried to build a couple of models, one that loads data from imdb dataset, and 
  the other that loads the labeled_data or any other data set. 

  The main model can be executed as follows:
  From the current directory, execute: `python3 model_builder.py`

  This will generate the model in .h5 format, which is consumed at later stages while predicting
  the movie score.


- `Step 4: Mandatory` The final step is to predict the review score of any movie using the any of the above models:

  To Analyze any movie by its name

  `python3 flick_screener.py --movie "Ford V Ferrari"`

  To analyze any movie by its partial name

  `python3 flick_screener.py --movie "Ford"`

  To analyze one particular tweet or any text for sentiment analysis

  `python3 flick_screener.py --tweet "I loved the movie!"`

### Examples
  - Once you have the security.json, just run
   `python3 flick_screener.py --m <movie_name>

    Example:
    python3 flick_screener.py -m sharknado
