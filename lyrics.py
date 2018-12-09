"""
Use Genius API to gather song lyrics based on song genre
Based on https://gist.github.com/imdkm/a60247b59ff1881fa4bb8846a9b44c96 and https://bigishdata.com/2016/09/27/getting-song-lyrics-from-geniuss-api-scraping/
"""

import csv
import urllib.request
import json
import requests

from bs4 import BeautifulSoup
from functools import wraps

# Requires user to have a Genius account and an access token, available at https://genius.com/signup_or_login
CLIENT_ACCESS_TOKEN = 'client_access_token'
BASE_URL = 'https://api.genius.com'


# Return an array of artist IDs for an array of artist names
def get_artist_ids(artists):
    artist_ids = []
    for artist in artists:
        find_id = get_data("search", {'q': artist})
        artist_id = None
        for hit in find_id["response"]["hits"]:
            if hit["result"]["primary_artist"]["name"].lower() == artist.lower():
                artist_id = (hit["result"]["primary_artist"]["id"])
        if not artist_id:
            print(artist)
        else:
            artist_ids.append(artist_id)
    return artist_ids


# Return array of song ids by all artists in a list of artist ids
def get_song_ids(artist_ids):
    song_ids = []
    for artist_id in artist_ids:
        current_page = 1
        next_page = True
        artist_songs = []
        song_data = []
        # Iterate through all pages of data on the artist
        while next_page:
            path = "artists/{}/songs/".format(artist_id)
            data = get_data(path=path, params={'page': current_page})

            page_songs = data['response']['songs']

            if page_songs:
                artist_songs += page_songs
                current_page += 1
            else:
                next_page = False

        # Add song id to array if primary artist matches user search, and if the url contains "-lyrics" (making the page likely to be an actual song)
        artist_songs = [song["id"] for song in artist_songs
                        if song["primary_artist"]["id"] == artist_id and "-lyrics" in song["url"]]
        song_ids.extend(artist_songs)
    return song_ids


# Scrape lyrics of a song
def get_lyrics(song_ids):
    # Authenticate API
    lyrics = ""
    for song_id in song_ids:
        song_web_path = "/songs/{}/".format(song_id)
        token = "Bearer {}".format(CLIENT_ACCESS_TOKEN)
        headers = {"Authorization": token}

        song_url = BASE_URL + song_web_path
        response = requests.get(song_url, headers=headers)
        json = response.json()
        path = json["response"]["song"]["path"]
        page_url = "http://genius.com" + path
        page = requests.get(page_url)
        html = BeautifulSoup(page.text, "html.parser")
        # remove script tags that they put in the middle of the lyrics
        [h.extract() for h in html('script')]
        # updated css where the lyrics are based in HTML
        song_lyrics = html.find("div", class_="lyrics").get_text()
        lyrics += (song_lyrics)
    return lyrics


# Remove problematic characters
def clean_lyrics(lyrics):
    chars = []
    brackets = False
    length = len(lyrics)
    for i in range(length):
        character = lyrics[i]
        if character in "([":       # Remove all content within parenthesis and square brackets
            brackets = True
        # Ensure RNN can process character, and eliminate double line breaks
        if not brackets and ord(lyrics[i]) < 128 and not (i != length - 1 and lyrics[i] == "\n" and lyrics[i + 1] == "\n"):
            chars.append(character)
        if character in ")]":
            brackets = False

    cleaned = "".join(chars)        # Concatenate characters into one string
    return cleaned


# Return JSON annotation of object
def get_data(path, params=None, headers=None):
    requrl = '/'.join([BASE_URL, path])
    token = "Bearer {}".format(CLIENT_ACCESS_TOKEN)
    if headers:
        headers['Authorization'] = token
    else:
        headers = {"Authorization": token}

    response = requests.get(url=requrl, params=params, headers=headers)
    response.raise_for_status()

    return response.json()


# Return all lyrics by a list of artists, given artists' names
def get_lyrics_from_artists(artists):
    artist_ids = get_artist_ids(artists)
    song_ids = get_song_ids(artist_ids)
    lyrics = get_lyrics(song_ids)
    return clean_lyrics(lyrics)


def main():
    # # Top 10, taken from Billboard top country artists in 2016: https://www.billboard.com/charts/year-end/2016/top-country-artists
    country_artists = ["Cole Swindell", "Jason Aldean", "Sam Hunt", "Keith Urban", "Luke Bryan",
                       "Carrie Underwood", "Thomas Rhett", "Florida Georgia Line", "Blake Shelton", "Chris Stapleton"]
    cleaned_country = get_lyrics_from_artists(country_artists)
    file = open("country.txt", "w")
    file.write(cleaned_country)
    file.close()

    # Top 10, taken from Billboard top R&B/Hip-Hop artists in 2016: https://www.billboard.com/charts/year-end/2016/top-r-and-b-hip-hop-artists
    hip_hop_artists = ["Chris Brown", "Fetty Wap", "Kevin Gates", "Bryson Tiller",
                       "Desiigner", "Future", "The Weeknd", "Rihanna", "Beyoncé", "Drake"]
    cleaned_hip_hop = get_lyrics_from_artists(hip_hop_artists)
    file = open("hiphop.txt", "w")
    file.write(cleaned_hip_hop)
    file.close()

    # Top 10 artists, taken from Billboard top Pop Songs Artists in 2016: https://www.billboard.com/charts/year-end/2016/pop-songs-artists
    pop_artists = ["Alessia Cara", "Daya", "Shawn Mendes", "Ariana Grande", "Drake", "Adele",
                   "Selena Gomez", "The Chainsmokers", "twenty one pilots", "Justin Bieber"]
    cleaned_pop = get_lyrics_from_artists(pop_artists)
    file = open("pop.txt", "w")
    file.write(cleaned_pop)
    file.close()


if __name__ == "__main__":
    main()