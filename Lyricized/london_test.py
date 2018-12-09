import rnn

# def main(train_model, generate_song, data_file, ckpt_file):
training_name = "test_train"
rnn.main(train_model=True, generate_song=True, data_file="London/LondonData/{}.txt".format(training_name),
         ckpt_file="London/LondonModels/{}/model.ckpt".format(training_name), song_file="London/LondonSongs/{}.txt".format(training_name))
