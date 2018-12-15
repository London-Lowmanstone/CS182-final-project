"""
Text generation using a Recurrent Neural Network (LSTM).
Based on https://github.com/cassandrakane/Lyricized
Based on https://github.com/spiglerg/RNN_Text_Generation_Tensorflow
"""
import tensorflow as tf
import numpy as np
import random
import time
import sys

class ModelNetwork:
    def __init__(self, in_size, lstm_size, num_layers, out_size, session, learning_rate=0.003, name="rnn"):
        self.scope = name

        self.in_size = in_size
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.out_size = out_size

        self.session = session

        self.learning_rate = tf.constant(learning_rate)

        # Last state of LSTM, used when running the network in TEST mode
        self.lstm_last_state = np.zeros((self.num_layers * 2 * self.lstm_size,))

        with tf.variable_scope(self.scope):
            ## (batch_size, timesteps, in_size)
            self.xinput = tf.placeholder(tf.float32,
                                         shape=(None, None, self.in_size),
                                         name="xinput")
            self.lstm_init_value = tf.placeholder(tf.float32,
                                                  shape=(None, self.num_layers *
                                                         2 * self.lstm_size),
                                                  name="lstm_init_value")

            # LSTM
            self.lstm_cells = [tf.contrib.rnn.BasicLSTMCell(
                self.lstm_size, forget_bias=1.0, state_is_tuple=False) for i in range(self.num_layers)]
            self.lstm = tf.contrib.rnn.MultiRNNCell(self.lstm_cells, state_is_tuple=False)

            # Iteratively compute output of recurrent network
            outputs, self.lstm_new_state = tf.nn.dynamic_rnn(
                self.lstm, self.xinput, initial_state=self.lstm_init_value, dtype=tf.float32)

            # Linear activation (FC layer on top of the LSTM net)
            self.rnn_out_W = tf.Variable(tf.random_normal(
                (self.lstm_size, self.out_size), stddev=0.01))
            self.rnn_out_B = tf.Variable(tf.random_normal(
                (self.out_size, ), stddev=0.01))

            outputs_reshaped = tf.reshape(outputs, [-1, self.lstm_size])
            network_output = (tf.matmul(outputs_reshaped, self.rnn_out_W) + self.rnn_out_B)

            batch_time_shape = tf.shape(outputs)
            self.final_outputs = tf.reshape(tf.nn.softmax(network_output),
                                            (batch_time_shape[0], batch_time_shape[1], self.out_size))

            # Training: provide target outputs for supervised training.
            self.y_batch = tf.placeholder(tf.float32, (None, None, self.out_size))
            y_batch_long = tf.reshape(self.y_batch, [-1, self.out_size])

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=network_output, labels=y_batch_long))
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, 0.9).minimize(self.cost)

    def run_step(self, x, init_zero_state=True):
        # Reset the initial state of the network.
        if init_zero_state:
            init_value = np.zeros((self.num_layers * 2 * self.lstm_size,))
        else:
            init_value = self.lstm_last_state

        out, next_lstm_state = self.session.run([self.final_outputs, self.lstm_new_state],
                                                feed_dict={self.xinput: [x], self.lstm_init_value: [init_value]})

        self.lstm_last_state = next_lstm_state[0]

        return out[0][0]

    # xbatch must be (batch_size, timesteps, input_size)
    # ybatch must be (batch_size, timesteps, output_size)
    def train_batch(self, xbatch, ybatch):
        init_value = np.zeros((xbatch.shape[0], self.num_layers * 2 * self.lstm_size))

        cost, _ = self.session.run([self.cost, self.train_op],
                                   feed_dict={self.xinput: xbatch, self.y_batch: ybatch, self.lstm_init_value: init_value})

        return cost


# Embed string to character-arrays -- it generates an array len(data) x len(vocab)
# Vocab is a list of elements
def embed_to_vocab(data_, vocab):
    data = np.zeros((len(data_), len(vocab)))

    cnt = 0
    for s in data_:
        v = [0.0] * len(vocab)
        v[vocab.index(s)] = 1.0
        data[cnt, :] = v
        cnt += 1

    return data


def decode_embed(array, vocab):
    return vocab[array.index(1)]


def main(train_model, generate_tweet, data_file, ckpt_file, tweet_file="tweet.txt"):
    """Train model/Generate tweet"""

    # Load the data
    data_ = ""
    with open(data_file, 'r') as f:
        data_ += f.read()
    data_ = data_.lower()

    # Convert to 1-hot coding
    vocab = sorted(list(set(data_)))
    data = embed_to_vocab(data_, vocab)

    # Hyperparameters
    in_size = out_size = len(vocab)
    lstm_size = 256
    num_layers = 2
    batch_size = 64
    time_steps = 100

    ITERATION_COUNT = 20000  # Number of training iterations
    LEN_TEST_TEXT = 200  # Number of words of text to generate after training the network
    SAVE_COUNT = 100 # Number of iterations before each save

    # Initialize the network
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    net = ModelNetwork(in_size=in_size,
                       lstm_size=lstm_size,
                       num_layers=num_layers,
                       out_size=out_size,
                       session=sess,
                       learning_rate=0.003,
                       name="char_rnn_network")

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    # Train the network
    if train_model:
        last_time = time.time()

        batch = np.zeros((batch_size, time_steps, in_size))
        batch_y = np.zeros((batch_size, time_steps, in_size))

        possible_batch_ids = range(data.shape[0] - time_steps - 1)
        for i in range(ITERATION_COUNT):
            print("Starting iteration {} / {}".format(i + 1, ITERATION_COUNT))
            # Sample time_steps consecutive samples from the dataset text file
            batch_id = random.sample(possible_batch_ids, batch_size)

            for j in range(time_steps):
                ind1 = [k + j for k in batch_id]
                ind2 = [k + j + 1 for k in batch_id]

                batch[:, j, :] = data[ind1, :]
                batch_y[:, j, :] = data[ind2, :]

            cst = net.train_batch(batch, batch_y)
            if i % SAVE_COUNT == 0:
                saver.save(sess, ckpt_file)
        
        if generate_tweet:
            generate_tweet(net, saver, ckpt_file)

def generate_tweet(net, saver, ckpt_file):
    # Generate LEN_TEST_TEXT words using the trained network
    if generate_tweet:
        print("Generating tweet...")
        saver.restore(sess, ckpt_file)

        TEST_PREFIX = " "
        for i in range(len(TEST_PREFIX)):
            out = net.run_step(embed_to_vocab(TEST_PREFIX[i], vocab), i == 0)

        gen_str = TEST_PREFIX
        word_count = 0
        tweet = open(tweet_file, "w+")
        while word_count < LEN_TEST_TEXT:
            # Sample character from the network according to the generated output probabilities
            element = np.random.choice(range(len(vocab)), p=out)
            gen_str += vocab[element]
            out = net.run_step(embed_to_vocab(vocab[element], vocab), False)
            if vocab[element] == " ":
                word_count += 1
        tweet.write(gen_str)
        tweet.close()
        print("Tweet saved at {}".format(tweet_file))
