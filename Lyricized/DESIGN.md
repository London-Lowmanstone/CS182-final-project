# DESIGN

### Overview
The goal of *Lyricized* was to collect a large compilation of song lyrics by a given genre and run it through a recurrent neural network (RNN) in order to generate a new set of lyrics that plausibly fit in that genre. This process is two-fold: first, we search [genius.com](https://genius.com/) for the top genre artists (taken from the 2016 Billboard charts), and use its API to obtain all of the lyrics written by those artists. Second, we run that data through an RNN, which generates a new set of artificial lyrics. We decided to display the final results on a web page for a more user-friendly experience.

For the sake of demonstration, much of this program is pre-processed. `lyrics.py` gathers large input files of lyrics, and the RNN is trained on each set of lyrics. We generated 50 unique songs per genre, and uploaded them as .txt files within `demo_songs`. When the user is on the website in demo mode, the program randomly selects one of the pre-generated songs to display because the CS50 IDE does not have the memory capacity to run the RNN live. If you wish to train models or generate songs, the program must be downloaded and run elsewhere.

The RNN proved fairly successful in its ability to train models and generate lyrics. After training for 20,000 iterations, `cost` (a measure of dissimilarity between the real and fake lyrics) fell from 4.00332 to 0.552482 for country, from 4.20527 to 1.12544 for hip hop, and from 4.00939 to 0.518021 for pop. When examining the generated lyrics, each song contained a majority of English words, only minor syntax errors, and an overall style that matched its genre.

### Genius API
This portion of the project collects all of the data on artists and their lyrics using the Genius API. Genius assigns unique IDs to every artist and every song, and stores lots of data about each. `lyrics.py` generates large .txt files with lots of lyrics from a given genre. It consists of several functions:
* `get_data` is a basic helper function, which given a unique song or artist id, returns a JSON array with all of the metadata about an artist or a song. Both `lookup_artist` and `lookup_songs` use this data to return more specific information. The following functions use `get_data`:
    * `get_artist_ids` takes an list of artist names, searches for the artists in Genius, and returns an list of the unique ID numbers of the artists, in the same order. If it cannot find the artist, it simply prints the artist's name.
    * `get_song_ids` takes an list of artist IDs, and returns an list of unique song IDs for every song with any of those artists listed as its "primary artist". The extra filtering in `lookup_songs` tries to account for the fact that some "songs" on Genius are actually other text pages. Most of the pages that contain "-lyrics" in their URLs are actually songs.
* `get_lyrics` takes an list of song IDs, and scrapes Genius for all of the lyrics in each song, Once all of the lyrics are concatenated into one long string, the string is cleaned by `clean_lyrics`. The RNN can only handle characters with ASCII values lower than 128, so this function removes any others. It also takes out all content inside square brackets or parenthesis. This tends to be text like "[VERSE 1]" or other non-lyrical text.
* The `main` function in `lyrics.py` hardcodes lyrics of the top country, hip-hop, and pop artists, based on Billboard's top charts. This list set to `get_lyrics_from_artists`, where it is processed by `get_artist_ids`, `get_song_ids`, and finally `get_lyrics`, such that the end result is a long string of all of the lyrics to all of the songs written by all of the artists. `main` then writes these strings into .txt files, all prepared for the RNN.

### Recurrent Neural Network (RNN)
This portion of the project generates lyrics based on a large dataset of lyrics in a given genre using a recurrent neural network (RNN). We used an RNN due to their proven effectiveness in text-generation (see [here](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) for more info). `rnn.py` contains:
* `ModelNetwork` (class) is the neural network with functions that ultimately aim to minimize `cost` (a measure of how well the RNN performs)
* `main` (function) trains the RNN (given the `train_model` argument is `True`) and generates `song.txt` (given the `generate_song` argument is `True`). `main` is called from `application.py`
    * Training: RNN goes through `ITERATION_COUNT` rounds of training (default = `20000`) with the following hyperparameters:
        * `lstm_size = 256` (size of the network)
	    * `num_layers = 2` (number of layers in the network)
	    * `batch_size = 64` (size of sample batches for training)
	    * `time_steps = 100` (number of consecutive samples)
	* Generation: Model generates approximately 200 words for `song.txt`
* `rnn.py` will not be run if `demo_mode = True` in `application.py` due to the CS50 IDE memory limit


### Web Design
*Lyricized* uses a very simple website, with only two templates. We wanted the focus of our project to be the API and the RNN, but we wanted a user-friendly way to display our work. The homepage, `index.html`, allows users to select a genre of music. Once the song is generated, the user is directed to `song.html`, which displays the lyrics. We used CSS and Bootstrap to lightly style the website, and modeled it after distribution code from problem set 7.