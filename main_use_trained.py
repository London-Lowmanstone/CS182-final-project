# use this if you've already trained a model

import rnn

# change these
training_name = "rep_tweets" # which LSTM you want to use (dem_tweets for Democrat)
how_many_tweets = 20 # how many tweets to create with the same start
output_file = "output.txt" # where the output tweets should go
prefix = " " # what the tweets should begin with

# leave these alone
data_folder = "data"
models_folder = "models"
data_file = "{}/{}.txt".format(data_folder, training_name)
ckpt_file ="{}/{}/model.ckpt".format(models_folder, training_name)

rnn.generate_tweets(how_many_tweets, output_file, data_file, ckpt_file, prefix)
