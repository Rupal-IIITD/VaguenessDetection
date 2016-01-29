# -*- coding: utf-8 -*-
"""
Created on Thu Apr 09 13:37:52 2015

@author: bargav_jayaraman
"""
import numpy
import theano
from theano import function
import theano.tensor as T
from collections import OrderedDict

rng = numpy.random

class RNN(object):
    def __init__(self, N, win, D, H, C, eta, lamb, we=None, wx=None, wh=None, bh=None, w=None, b=None, h0=None):
        if we == None:
            self.we = theano.shared(rng.uniform(-0.1, 0.1, (N+1, D)).astype(theano.config.floatX), name="we")
        else:
            self.we = we
        if wx == None:
            self.wx = theano.shared(rng.uniform(-numpy.sqrt(6./(win*D+H)), numpy.sqrt(6./(win*D+H)), (win*D, H)).astype(theano.config.floatX), name="wx")
        else:
            self.wx = wx
        if wh == None:
            self.wh = theano.shared(rng.uniform(-numpy.sqrt(6./(H+H)), numpy.sqrt(6./(H+H)), (H, H)).astype(theano.config.floatX), name="wh")
        else:
            self.wh = wh
        if bh == None:
            self.bh = theano.shared(numpy.zeros((H,), dtype=theano.config.floatX), name="bh")
        else:
            self.bh = bh
        if w == None:
            self.w = theano.shared(rng.uniform(-numpy.sqrt(6./(H+C)), numpy.sqrt(6./(H+C)), (H, C)).astype(theano.config.floatX), name="w")
        else:
            self.w = w
        if b == None:
            self.b = theano.shared(numpy.zeros((C,), dtype=theano.config.floatX), name="b")
        else:
            self.b = b
        if h0 == None:
            self.h0 = theano.shared(numpy.zeros(H, dtype=theano.config.floatX), name="h0")
        else:
            self.h0 = h0
        
        self.params = [self.we, self.wx, self.wh, self.bh, self.w, self.b, self.h0]
        idxs = T.imatrix()
        x = self.we[idxs].reshape((idxs.shape[0], D*win))
        y = T.ivector('y')
                
        def recurrence(x_t, h_tm1):
            h_t = T.tanh(T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence,
                        sequences=x,
                        outputs_info=[self.h0, None],
                        n_steps=x.shape[0])

        p_y_given_x = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x, axis=1)
        
        nll = -T.mean(T.log(p_y_given_x)[T.arange(x.shape[0]), y]) \
        + lamb * (self.wx ** 2).sum() \
        + lamb * (self.wh ** 2).sum() \
        + lamb * (self.w ** 2).sum()
        gradients = T.grad(nll, self.params)
        updates = OrderedDict((p, p - eta*g) for p, g in zip(self.params, gradients))

        self.test = function(inputs = [idxs],
                           outputs = y_pred)

        self.train = function(inputs = [idxs, y],
                        outputs = nll,
                        updates = updates)

    def get_params(self):
        return self.params
    