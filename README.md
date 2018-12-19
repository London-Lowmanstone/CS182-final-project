# CS182 Final Project 2018 - LSTM Portion

Use Python3 with the requirements listed in `requirements.txt` installed to run the files named `main_rep.py` and `main_dem.py`. This will display how many iterations are left in the training for the dataset on Republican tweets and Democrat tweets respectively. Training for 20,000 iterations (the default) on both datasets with only a CPU generally takes about 24 hours.

Once training is complete, you can generate tweets by running `main_use_trained.py` in order to create tweets and save them to a file. You can choose the start of the tweet and let the LSTM finish it. The `training name` variable should be the same in both the training main program and the tweet generation main program. You can train and generate text on your own files by changing
that variable and putting a text file into the `data` folder with that name and adding in `.txt`.

All the code for the LSTM is inside the `rnn.py` file.

Feel free to contact me via email if there are any issues.
