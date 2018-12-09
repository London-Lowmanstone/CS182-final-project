from flask import Flask, flash, redirect, render_template, request, session
from flask_session import Session
from tempfile import mkdtemp
from random import sample, randint
import rnn


# determines website mode (does not run RNN in demo mode)
DEMO_MODE = True

# determines whether or not to train models
country_model_trained = True
hiphop_model_trained = True
pop_model_trained = True


# Configure application
app = Flask(__name__)


# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    """Allow user to select a genre"""
    if request.method == "POST":
        genre = request.form.get("genre")
        song_file = ""
        if DEMO_MODE:
            # randomly choose pre-generated song in genre
            song_file = "demo_songs/%s/song%s.txt" % (genre, str(randint(1, 50)))
        else:
            # check whether genre model should be trained
            model_trained = (genre == "country" and not country_model_trained) or (
                genre == "hiphop" and not hiphop_model_trained) or (genre == "pop" and not pop_model_trained)
            # train model/generate song
            rnn.main(not model_trained, True,
                     "data/%s.txt" % (genre), "models/%s/model.ckpt" % (genre))
            song_file = "song.txt"
        # get/clean generated song lyrics
        lyrics = clean_generated_lyrics(song_file)
        return render_template("song.html", text=lyrics)

    else:
        return render_template("index.html")


def clean_generated_lyrics(song_file):
    """Clean generated lyrics (remove whitespace, capitalize, etc)"""
    song = open(song_file, "r").readlines()
    lyrics = ""
    for line in song:
        line = line.lstrip().capitalize()
        lyrics += line
    return lyrics