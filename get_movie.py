#!/usr/bin/env python

import requests
import sys
import urllib.parse

# Syntax
# python get_movie.py <movie name>"
# Example: python get_movie.py "ford v ferrari"
#
# Partial Matches Work too
# Example2: python get_movie.py "ford"
#
# When Multiple movies match the same name pattern, the latest film among them is chosen

# TODO: encrypt API key
omdb_search_url = "http://www.omdbapi.com/?s={}&apikey=334c80ae"
omdb_get_by_id_url = "http://www.omdbapi.com/?i={}&apikey=334c80ae"


def search_movie_by_name(given_name):
    encoded_name = urllib.parse.quote(given_name)
    movie = {}
    resp = requests.get(omdb_search_url.format(encoded_name))
    if resp is None or resp.status_code != 200 or resp.json() is None or resp.json().get('Search',None) is None or resp.json().get('Error',None) is not None:
        # This means something went wrong.
        raise Exception('Unable to Find the movie ')
    for result in resp.json()['Search']:
        year = int(result['Year'])
        if movie.get('year', 0) < year:
            movie['title'] = result['Title']
            movie['year'] = year
            movie['imdb_id'] = result['imdbID']
    if movie.get('imdb_id', None) is None:
        raise Exception('Unable to Find the movie ')
    return movie


def get_movie_score_by_id(imdb_id):
    movie = {}
    if imdb_id is None or not imdb_id:
        raise Exception("Invalid Movie Details")
    resp = requests.get(omdb_get_by_id_url.format(imdb_id))
    if resp is None or resp.status_code != 200 or resp.json() is None or resp.json().get('Error',None) is not None:
        raise Exception('Unable to Find the movie ')
    details = resp.json()
    movie['genre'] = details['Genre']
    for rating in details['Ratings']:
        if rating['Source'] == "Internet Movie Database":
            movie['imdb_rating'] = float(rating['Value'].replace("/10", ""))
        if rating['Source'] == "Metacritic":
            movie['metascore'] = int(rating['Value'].replace("/100", ""))
    return movie


def get_movie(movie_name):
    movie=search_movie_by_name(movie_name)
    movie.update(get_movie_score_by_id(movie['imdb_id']))
    return movie


def main():
    if len(sys.argv) != 2:
        raise Exception("Please use right syntax:\n \tpython get_movie.py \"<movie name>\" ")
    given_name = sys.argv[1].strip()
    print(get_movie(given_name))


if __name__ == '__main__':
    main()
