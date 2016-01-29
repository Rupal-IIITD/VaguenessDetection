# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 09:28:32 2015

@author: bargav_jayaraman
"""

import numpy
from numpy import *
import theano
from theano import function
import theano.tensor as T

rng = numpy.random

class AE(object):
    def __init__(self, N1, N2, D, eta, forward=True, we1=None, we2=None, lamda=4):
        if we1 == None:
            self.we1 = theano.shared(rng.uniform(-0.1, 0.1, (N1, D)).astype(theano.config.floatX), name="we1")
        else:
            self.we1 = we1
        if we2 == None:
            self.we2 = theano.shared(rng.uniform(-0.1, 0.1, (N2, D)).astype(theano.config.floatX), name="we2")
        else:
            self.we2 = we2
        self.b = theano.shared(numpy.zeros((D,), dtype=theano.config.floatX), name="b")
        self.b_prime = theano.shared(numpy.zeros((N1+N2,), dtype=theano.config.floatX), name="b_prime")

        self.we = T.concatenate([self.we1, self.we2])
        self.we_prime = self.we.T
        if forward:
            self.params = [self.we2, self.b, self.b_prime]
        else:
            self.params = [self.we1, self.b, self.b_prime]
        x = T.matrix('x')
        
        y1 = (T.dot(x[:,N1:N1+N2], self.we2) + self.b)
        z1 = (T.dot(T.nnet.sigmoid(y1), self.we_prime) + self.b_prime)

        y2 = (T.dot(x[:,0:N1], self.we1) + self.b)
        z2 = (T.dot(T.nnet.sigmoid(y2), self.we_prime) + self.b_prime)
            
        y3 = T.nnet.sigmoid(T.dot(x, self.we) + self.b)
        z3 = (T.dot(y3, self.we_prime) + self.b_prime)

        cor = list()

        for i in range(0,D):
            x1 = y1[:,i] - (ones(20)*(T.sum(y1[:,i])/20))
            x2 = y2[:,i] - (ones(20)*(T.sum(y2[:,i])/20))
            nr = T.sum(x1 * x2) / (T.sqrt(T.sum(x1 * x1))*T.sqrt(T.sum(x2 * x2)))
            cor.append(-nr)
            
        L1 = - T.sum(x * T.log(T.nnet.sigmoid(z1)) + (1 - x) * T.log(1 - T.nnet.sigmoid(z1)), axis=1)
        L2 = - T.sum(x * T.log(T.nnet.sigmoid(z2)) + (1 - x) * T.log(1 - T.nnet.sigmoid(z2)), axis=1)
        L3 = - T.sum(x * T.log(T.nnet.sigmoid(z3)) + (1 - x) * T.log(1 - T.nnet.sigmoid(z3)), axis=1)
        L4 =  T.sum(cor)
        L = L1 + L2 + L3 + (lamda * L4) + 100

        cost = T.mean(L)

        gparams = T.grad(cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - eta*gparam))
        
        self.train = function(inputs=[x],
                              outputs=cost,
                              updates=updates)

    def get_we2(self):
        return self.we2
    
    def get_we1(self):
        return self.we1