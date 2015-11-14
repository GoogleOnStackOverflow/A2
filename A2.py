# this is the 1st Assignment of Machine Learning
__docformat__ = 'restructedtext en'

import os
import sys
import timeit
import cPickle
import gzip
from itertools import izip

import numpy as np

import theano
import theano.tensor as T

import scipy.io
import csv

# Define Sigmoid function
def sigmoid(z):
	return 1/(1+T.exp(-z))

# Define RELU function 
def relu (x):
	return T.switch(x<0,0,x)


# Number of units in the hidden (recurrent) layer
N_HIDDEN = 100
# input
N_INPUT = 48
# output
N_OUTPUT = 1

Train = scipy.io.loadmat('Train_p.mat')
Test = scipy.io.loadmat('Test_p.mat') 
x_seq=Train['Train_P']
x_seq_test=Test['Test_P']

y_hat = np.array(list(csv.reader(open('./y_int.ark', 'rb'),delimiter=' ')))
y_hat = y_hat.flatten()
y_hat = y_hat.astype('int32')

print x_seq.shape
print x_seq_test.shape
print y_hat.shape

"""
# Define Softmax function
def softmax (x):
	x_a = np.array(x)
    e_x = np.exp(x_a - np.max(x_a))
    out = e_x / e_x.sum()
    return out
"""

# Define Update function using momentum
def gradient_updates_momentum(cost, params, learning_rate, momentum):
    '''
    Compute updates for gradient descent with momentum
    
    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - momentum : float
            Momentum parameter, should be at least 0 (standard gradient descent) and less than 1
   
    :returns:
        updates : list
            List of updates, one for each parameter
    '''
    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        # For each parameter, we'll create a param_update shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that when updating param_update, we are using its old value and also the new gradient step.
        updates.append((param, param - learning_rate*param_update))
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
    return updates


Wi = theano.shared( np.random.randn(N_INPUT,N_HIDDEN) )
bh = theano.shared( np.zeros(N_HIDDEN) )
Wo = theano.shared( np.random.randn(N_HIDDEN,N_OUTPUT) )
bo = theano.shared( np.zeros(N_OUTPUT) )
Wh = theano.shared( np.random.randn(N_HIDDEN,N_HIDDEN) )
a_0 = theano.shared(np.zeros( (x_seq.shape[0],N_HIDDEN) ))
y_0 = theano.shared(np.zeros(x_seq.shape[0]))
parameters = [Wi,bh,Wo,bo,Wh]

print Wi.type
print bh.type
print Wo.type
print bo.type
print Wh.type

"""
def step (x_t, a_tm1):
	a_t = sigmoid( T.dot(x_t,Wi) + T.dot(a_tm1,Wh) + bh )
	y_t = T.nnet.softmax( T.dot(a_tm1, Wo) + bo)
	return a_t,y_t

a_seq,_ =theano.scan(
		step,
		sequences = x_seq,
		outputs_info = a_0,
		truncate_gradient=-1
	)  

"""
def step(z_t,a_tm1):
	# z_t: N_BATCH X N_HIDDEN, a_tm1: N_BATCH X N_HIDDEN
        return sigmoid( z_t + T.dot(a_tm1,Wh) + bh )
	# return: N_BATCH X N_HIDDEN

z_seq = T.dot(x_seq,Wi) 
# x_seq_batch: length X N_BATCH X 2
# Wi: 2 X N_HIDDEN
# z_seq_batch: length X N_BATCH X N_HIDDEN

a_seq,_ = theano.scan(
                        step,
                        sequences = z_seq,
				#sequences: length X N_BATCH X N_HIDDEN
                        outputs_info = a_0,
                        truncate_gradient=-1
                )

y_seq = T.dot(a_seq, Wo) + bo.dimshuffle('x',0)

cost = T.sum( (y_seq - y_hat) ** 2 )
gradients = T.grad(cost,parameters)

def MyUpdate(parameters,gradients):
	mu = np.float32(0.001)
	parameters_updates = [(p,p - mu * g) for p,g in izip(parameters,gradients) ] 
	return parameters_updates

rnn_train = theano.function(
	inputs=[x_seq,y_hat],
	outputs=cost,
	updates=MyUpdate(parameters,gradients)
)

for i in range(10000000):
        print "iteration:", i, "cost:",  rnn_train(x_seq,y_hat)

