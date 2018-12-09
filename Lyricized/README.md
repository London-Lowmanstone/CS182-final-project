# README

### About
*Lyricized* is a website that generates song lyrics in a given genre based on the songs from popular artists within said genre. The Python-based website contains a "demo mode" that runs on the CS50 IDE using Flask, and the program can be run in its full power elsewhere. The following submission contains many files, some of which were used to acquire and hard-code data, while others are necessary for running the website. On the website, the user has the option to pick one of three music genres: country, hip-hop, or pop. They will be directed to a page of lyrics to a plausible (but fake) song in that genre. In demo mode, the webpage randomly chooses lyrics from a pre-generated selection of songs. If not in demo mode, a user could train the RNN and generate songs live.

### Submission Contents
* `data/` contains text files of original lyrics scraped from Genius
* `demo_songs/` contains 50 pre-generated songs per genre
* `models` contains trained models (ckpt files) for each genre based on `data/`
* `static/` contains the CSS stylesheet
* `templates/` contains HTML templates for the index and song pages
* `application.py` runs the web application using Flask
* `lyrics.py` scrapes Genius for the lyrics of top artists, given their names
* `requirements.txt` contains required dependencies for running program
* `rnn.py` contains a recurrent neural network that trains models and generates lyrics based on said models (unused in demo mode)
* `song.txt` contains the generated song output from `rnn.py` (unused in demo mode)

### Instructions for Use
* Demo Mode (Intended to be run on CS50 IDE)
    * Set `DEMO_MODE = True` in `application.py`
    * Execute `flask run` in terminal window to start website
    * Select genre from home page, view generated lyrics (randomly selected from 50 pre-generated songs)
* Not Demo Mode (Cannot be run on CS50 IDE)
    * Set `DEMO_MODE = False` in `application.py`
    * Training models: set `country_model_trained = False`, or `hiphop_model_trained = False`, or `pop_model_trained = False` in order to retrain models in specific genre.
        * Running the RNN will require the installation of TensorFlow (see [here](https://www.tensorflow.org/install/) for more info).
        * Models perform better given more data and more training iterations
    * Generating songs: generated lyrics using trained models will be stored in `song.txt`
    * Execute `flask run` in terminal window to start website
    * Select genre from home page, view generated lyrics
* Other Notes
    * Running `lyrics.py`:
        * Set `CLIENT_ACCESS_TOKEN` to your Genius API access token (available [here](https://genius.com/signup_or_login))
        * Execute `python lyrics.py` in terminal window in order to gather all lyrics for top ten artists in each genre (country, hip hop, pop)
            * Top ten artists are hardcoded in `country_artists`, `hip_hop_artists`, and `pop_artists` arrays, and can be updated accordingly
        * `lyrics.py` has already been executed; lyric files are available in `data/` folder

### Acknowledgements
Much of the code is inspired by several other projects, without which this project would not be possible.

In `lyrics.py`:
* `get_data`, `get_artists`, and `get_songs` were all inspired by [this GitHub post](https://gist.github.com/imdkm/a60247b59ff1881fa4bb8846a9b44c96)
* The scraping system in `get_lyrics` is based on code from [this website](https://bigishdata.com/2016/09/27/getting-song-lyrics-from-geniuss-api-scraping/).

In `rnn.py`:
* The code is heavily inspired by [this GitHub post](https://github.com/spiglerg/RNN_Text_Generation_Tensorflow)
    * We learned about RNN processing [here](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

The web application is based on the distribution code from Problem Set 7.

Additional thanks to our TFs, Derek Wang and Vojta Drmota, and the entire CS50 staff at Harvard.

*Created by Cassandra Kane '21 and Samantha Hung '20 for Computer Science 50 at Harvard College, Fall 2017.*