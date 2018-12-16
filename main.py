import rnn

# def main(train_model, generate_song, data_file, ckpt_file):
training_name = "rep_tweets"

data_folder = "data"
models_folder = "models"
new_tweets_folder = "tweets"
data_file = "{}/{}.txt".format(data_folder, training_name)
ckpt_file ="{}/{}/model.ckpt".format(models_folder, training_name)
rnn.main(iterations=20000, should_generate_tweet=True,
         data_file=data_file,
         ckpt_file=ckpt_file,
         tweet_file="{}/{}.txt".format(new_tweets_folder, training_name))

# Use one or the other, but not both
# rnn.generate_tweet(5, "quick_tweet.txt", data_file, ckpt_file)
