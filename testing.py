# from http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/

# Update: PyTorch Implementation of the same notebook available here.
#
# A short while ago Google open-sourced TensorFlow, a library designed to allow easy computations on graphs. The main applications are targeted for deep learning, as neural networks are represented as graphs. I’ve spent a few days reading their APIs and tutorials and I like what I see. Although other libraries offer similar features such as GPU computations and symbolic differentiation, the cleanliness of the API, and familiarity of the IPython stack makes it appealing to use.
#
# In this post I try to use TensorFlow to implement the classic Mixture Density Networks (Bishop ’94) model. I have implemented MDN’s before in an earlier blog post. This is my first attempt at learning to use TensorFlow, and there are probably much better ways to do many things, so let me know in the comment section!
#
# Simple Data Fitting with TensorFlow
# To get started, let’s try to quickly build a neural network to fit some fake data. As neural nets of even one hidden layer can be universal function approximators, we can see if we can train a simple neural network to fit a noisy sinusoidal data, like this (  \epsilonis just standard gaussian random noise):
#
# y=7.0 \sin( 0.75 x) + 0.5 x + \epsilon



import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math


# After importing the libraries, we generate the sinusoidal data we will train a neural net to fit later:



NSAMPLE = 1000
x_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
r_data = np.float32(np.random.normal(size=(NSAMPLE,1)))
y_data = np.float32(np.sin(0.75*x_data)*7.0+x_data*0.5+r_data*1.0)

plt.figure(figsize=(8, 8))
plot_out = plt.plot(x_data,y_data,'ro',alpha=0.3)
plt.show()


# TensorFlow uses place holders as variables that will eventually hold data, to do symbolic computations on the graph later on.



x = tf.placeholder(dtype=tf.float32, shape=[None,1])
y = tf.placeholder(dtype=tf.float32, shape=[None,1])


# We will define this simple neural network one-hidden layer and 20 nodes:
#
#  Y = W_{out} \tanh( W X + b) + b_{out}



NHIDDEN = 20
W = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=1.0, dtype=tf.float32))
b = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=1.0, dtype=tf.float32))

W_out = tf.Variable(tf.random_normal([NHIDDEN,1], stddev=1.0, dtype=tf.float32))
b_out = tf.Variable(tf.random_normal([1,1], stddev=1.0, dtype=tf.float32))

hidden_layer = tf.nn.tanh(tf.matmul(x, W) + b)
y_out = tf.matmul(hidden_layer,W_out) + b_out


# We can define a loss function as the sum of square error of the output vs the data (we can add regularisation if we want).



lossfunc = tf.nn.l2_loss(y_out-y);


# We will also define a training operator that will tell TensorFlow to minimise the loss function later. With just a line, we can use the fancy RMSProp gradient descent optimisation method.



train_op = tf.train.RMSPropOptimizer(learning_rate=0.1, decay=0.8).minimize(lossfunc)


# To start using TensorFlow to compute things, we have to define a session object. In an IPython shell, we use InteractiveSession. Afterwards, we need to run a command to initialise all variables, where the computation graph will also be built inside TensorFlow.



sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())


# We will run gradient descent for 1000 times to minimise the loss function with the data fed in via a dictionary. As the dataset is not large, we won’t use mini batches. After the below is run, the weight and bias parameters will be auto stored in their respective tf.Variable objects.



NEPOCH = 1000
for i in range(NEPOCH):
  sess.run(train_op,feed_dict={x: x_data, y: y_data})


# What I like about the API is that we can simply use sess.run() again to generate data from any operator or node within the network. So after the training is finished, we can just use the trained model, and another call to sess.run() to generate the predictions, and plot the predicted data vs the training data set.
#
# We should also close() the session afterwards to free resources when we are done with this exercise.



x_test = np.float32(np.arange(-10.5,10.5,0.1))
x_test = x_test.reshape(x_test.size,1)
y_test = sess.run(y_out,feed_dict={x: x_test})

plt.figure(figsize=(8, 8))
plt.plot(x_data,y_data,'ro', x_test,y_test,'bo',alpha=0.3)
plt.show()

sess.close()

# We see that the neural network can fit this sinusoidal data quite well, as expected. However, this type of fitting method only works well when the function we want to approximate with the neural net is a one-to-one, or many-to-one function. Take for example, if we invert the training data:
#
# x=7.0 \sin( 0.75 y) + 0.5 y+ \epsilon



temp_data = x_data
x_data = y_data
y_data = temp_data

plt.figure(figsize=(8, 8))
plot_out = plt.plot(x_data,y_data,'ro',alpha=0.3)
plt.show()

# If we were to use the same method to fit this inverted data, obviously it wouldn’t work well, and we would expect to see a neural network trained to fit only to the square mean of the data.



sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in range(NEPOCH):
  sess.run(train_op,feed_dict={x: x_data, y: y_data})

x_test = np.float32(np.arange(-10.5,10.5,0.1))
x_test = x_test.reshape(x_test.size,1)
y_test = sess.run(y_out,feed_dict={x: x_test})

plt.figure(figsize=(8, 8))
plt.plot(x_data,y_data,'ro', x_test,y_test,'bo',alpha=0.3)
plt.show()

sess.close()

# Our current model only predicts one output value for each input, so this approach will fail miserably. What we want is a model that has the capacity to predict a range of different output values for each input. In the next section we implement a Mixture Density Network (MDN) to do achieve this task.
#
# Mixture density networks
# Mixture Density Networks (MDNs), developed by Christopher Bishop in the 90’s, attempt to address this problem. The approach is rather to have the network predict a single output value, the network is to predict an entire probability distribution for the output. This concept is quite powerful, and can be employed many current areas of machine learning research. It also allows us to calculate some sort of confidence factor in the predictions that the network is making.
#
# The inverse sinusoidal data we chose is not just for a toy problem, as there are applications in the field of robotics, for example, where we want to determine which angle we need to move the robot arm to achieve a target location. MDNs can also used to model handwriting, where the next stroke is drawn from a probability distribution of multiple possibilities, rather than sticking to one prediction.
#
# Bishop’s implementation of MDNs will predict a class of probability distributions called Mixture Gaussian distributions, where the output value is modelled as a sum of many gaussian random values, each with different means and standard deviations. So for each input x, we will predict a probability distribution function (pdf) of  P(Y = y | X = x)that is a probability weighted sum of smaller gaussian probability distributions.
#
# P(Y = y | X = x) = \sum_{k=0}^{K-1} \Pi_{k}(x) \phi(y, \mu_{k}(x), \sigma_{k}(x)), \sum_{k=0}^{K-1} \Pi_{k}(x) = 1
#
# Each of the parameters  \Pi_{k}(x), \mu_{k}(x), \sigma_{k}(x)of the distribution will be determined by the neural network, as a function of the input x. There is a restriction that the sum of  \Pi_{k}(x)add up to one, to ensure that the pdf integrates to 100%. In addition,  \sigma_{k}(x)must be strictly positive.
#
# In our implementation, we will use a neural network of one hidden later with 24 nodes, and also generate 24 mixtures, hence there will be 72 actual outputs of our neural network of a single input. Our definition will be split into 2 parts:
#
#  Z = W_{o} \tanh( W_{h} X + b_{h}) + b_{o}
#
# Zis a vector of 72 values that will be then splitup into three equal parts, Z_{0\rightarrow23}, Z_{24\rightarrow43, and Z_{44\rightarrow71}
#
# The parameters of the pdf will be defined as below to satisfy the earlier conditions:
#
# \Pi_{k} = \frac{\exp(Z_{k})}{\sum_{i=0}^{23} exp(Z_{i})}, \sigma = \exp(Z_{24\rightarrow43}), \mu = Z_{44\rightarrow71}
#
# \Pi_{k}are essentially put into a softmax operator to ensure that the sum adds to one, and that each mixture probability is positive. Each  \sigma_{k}will also be positive due to the exponential operator. It gets deeper than this though! In Bishop’s paper, he notes that the softmax and exponential terms have some theoretical interpretations from a Bayesian framework way of looking at probability. Note that in the actual softmax code, the largest value will divide both numerator and denominator to avoid exp operator easily blowing up. In TensorFlow, the network is described below:



NHIDDEN = 24
STDEV = 0.5
KMIX = 24 # number of mixtures
NOUT = KMIX * 3 # pi, mu, stdev

x = tf.placeholder(dtype=tf.float32, shape=[None,1], name="x")
y = tf.placeholder(dtype=tf.float32, shape=[None,1], name="y")

Wh = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=STDEV, dtype=tf.float32))
bh = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=STDEV, dtype=tf.float32))

Wo = tf.Variable(tf.random_normal([NHIDDEN,NOUT], stddev=STDEV, dtype=tf.float32))
bo = tf.Variable(tf.random_normal([1,NOUT], stddev=STDEV, dtype=tf.float32))

hidden_layer = tf.nn.tanh(tf.matmul(x, Wh) + bh)
output = tf.matmul(hidden_layer,Wo) + bo

def get_mixture_coef(output):
  out_pi = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")
  out_sigma = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")
  out_mu = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")

  # needed arguments to be flipped here
  # https://stackoverflow.com/a/41842564 https://github.com/tensorflow/tensorflow/blob/64edd34ce69b4a8033af5d217cb8894105297d8a/RELEASE.md
  out_pi, out_sigma, out_mu = tf.split(output, 3, 1)

  max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
  # tf.subtract not tf.sub
  out_pi = tf.subtract(out_pi, max_pi)

  out_pi = tf.exp(out_pi)

  # tf.reciprocal not tf.inv
  normalize_pi = tf.reciprocal(tf.reduce_sum(out_pi, 1, keep_dims=True))
  # tf.multiply not tf.mul
  out_pi = tf.multiply(normalize_pi, out_pi)

  out_sigma = tf.exp(out_sigma)

  return out_pi, out_sigma, out_mu

out_pi, out_sigma, out_mu = get_mixture_coef(output)


# Let’s define the inverted data we want to train our MDN to predict later. As this is a more involved prediction task, I used a higher number of samples compared to the simple data fitting task earlier.



NSAMPLE = 2500

y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
r_data = np.float32(np.random.normal(size=(NSAMPLE,1))) # random noise
x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)

plt.figure(figsize=(8, 8))
plt.plot(x_data,y_data,'ro', alpha=0.3)
plt.show()

# We cannot simply use the min square error L2 lost function in this task the output is an entire description of the probability distribution. A more suitable loss function is to minimise the logarithm of the likelihood of the distribution vs the training data:
#
# CostFunction(y | x) = -\log[ \sum_{k}^K \Pi_{k}(x) \phi(y, \mu(x), \sigma(x)) ]
#
# So for every  (x, y)point in the training data set, we can compute a cost function based on the predicted distribution versus the actual points, and then attempt the minimise the sum of all the costs combined. To those who are familiar with logistic regression and cross entropy minimisation of softmax, this is a similar approach, but with non-discretised states.
#
# We implement this cost function in TensorFlow below as an operation:



oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi) # normalisation factor for gaussian, not needed.
def tf_normal(y, mu, sigma):
  result = tf.sub(y, mu)
  result = tf.mul(result,tf.inv(sigma))
  result = -tf.square(result)/2
  return tf.mul(tf.exp(result),tf.inv(sigma))*oneDivSqrtTwoPI

def get_lossfunc(out_pi, out_sigma, out_mu, y):
  result = tf_normal(y, out_mu, out_sigma)
  result = tf.mul(result, out_pi)
  result = tf.reduce_sum(result, 1, keep_dims=True)
  result = -tf.log(result)
  return tf.reduce_mean(result)

lossfunc = get_lossfunc(out_pi, out_sigma, out_mu, y)
train_op = tf.train.AdamOptimizer().minimize(lossfunc)


# We will train the model below, while collecting the progress of the loss function over training iterations. TensorFlow offers some useful tools to visualise training data progress but we didn’t use them here.



sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

NEPOCH = 10000
loss = np.zeros(NEPOCH) # store the training progress here.
for i in range(NEPOCH):
  sess.run(train_op,feed_dict={x: x_data, y: y_data})
  loss[i] = sess.run(lossfunc, feed_dict={x: x_data, y: y_data})


# As a side note, the impressive thing is that TensorFlow automatically calculated the gradients of the log likelihood cost functions and applied those gradients in the optimisation. For this problem, actually there are very optimised gradient formulas available (see derivations in Bishop’s original paper, equations 33-39), and I highly doubt the gradient formula TensorFlow calculated automatically will be as optimised and elegant, so there would be room for performance improvements by building a custom operator into TensorFlow with the pre-optimised gradient formulas for this loss function, like they have done for the cross entropy loss function. I had implemented before the optimised closed-form gradient formulas with all the numerical gradient testing that came with it – if you want to implement it, please make sure you do the gradient testing! It is difficult to get right the first time.
#
# Let’s plot the progress of the training, and see if the cost decreases over iterations:



plt.figure(figsize=(8, 8))
plt.plot(np.arange(100, NEPOCH,1), loss[100:], 'r-')
plt.show()

# We found that it more or less stopped improving after 6000 iterations or so. The next thing we want to do, is to have the model generate distributions for us, say, across a bunch of points along the x-axis, and then for each distribution, draw 10 points randomly from that distribution, to produce ensembles of generated data on the y-axis. This gives us a feel of whether the pdf generated matches the training data.
#
# To sample a mixed gaussian distribution, we randomly select which distribution based on the set of  \Pi_{k}probabilities, and then proceed to draw the point based off the  k^{th}gaussian distribution.



x_test = np.float32(np.arange(-15,15,0.1))
NTEST = x_test.size
x_test = x_test.reshape(NTEST,1) # needs to be a matrix, not a vector

def get_pi_idx(x, pdf):
  N = pdf.size
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if (accumulate >= x):
      return i
  print('error with sampling ensemble')
  return -1

def generate_ensemble(out_pi, out_mu, out_sigma, M = 10):
  NTEST = x_test.size
  result = np.random.rand(NTEST, M) # initially random [0, 1]
  rn = np.random.randn(NTEST, M) # normal random matrix (0.0, 1.0)
  mu = 0
  std = 0
  idx = 0

  # transforms result into random ensembles
  for j in range(0, M):
    for i in range(0, NTEST):
      idx = get_pi_idx(result[i, j], out_pi[i])
      mu = out_mu[i, idx]
      std = out_sigma[i, idx]
      result[i, j] = mu + rn[i, j]*std
  return result


# Let’s see how the generated data looks like:



out_pi_test, out_sigma_test, out_mu_test = sess.run(get_mixture_coef(output), feed_dict={x: x_test})

y_test = generate_ensemble(out_pi_test, out_mu_test, out_sigma_test)

plt.figure(figsize=(8, 8))
plt.plot(x_data,y_data,'ro', x_test,y_test,'bo',alpha=0.3)
plt.show()
