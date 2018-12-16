import rnn

# def main(train_model, generate_song, data_file, ckpt_file):
training_name = "test_tweets"

data_folder = "data"
models_folder = "models"
new_tweets_folder = "tweets"
rnn.main(train_model=True, generate_tweet=True,
         data_file="{}/{}.txt".format(data_folder, training_name),
         ckpt_file="{}/{}/model.ckpt".format(models_folder, training_name),
         song_file="{}/{}.txt".format(new_tweets_folder, training_name))
