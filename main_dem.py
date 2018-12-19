# use this to train a model on Democratic tweets

import rnn

# change this to train on a different dataset
training_name = "dem_tweets"

# leave these alone
data_folder = "data"
models_folder = "models"
new_tweets_folder = "tweets"
data_file = "{}/{}.txt".format(data_folder, training_name)
ckpt_file ="{}/{}/model.ckpt".format(models_folder, training_name)
rnn.train_model(iterations=20000,
                data_file=data_file,
                ckpt_file=ckpt_file)
