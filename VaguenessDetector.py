# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:48:50 2016

@author: bargav_jayaraman

Description: This script is the Trained Vagueness Detection Tool which can be used to detect vagueness in any 'Target' language, provided the required trained parameters for the 'Target' language are present. For Spanish and Portuguese, the trained parameters are already provided in the 'trained params' folder of the respective language folders. Users can also run the 'VaguenessDetectorTrain.py' script to get the trained parameters.

The script below detects the vagueness on sample test set. Users can supply their own data set instead.
"""

import time
import cPickle

from RNN import RNN

win = 7
D = 50
C = 2
H = 150
eta = 0.1
lamb = 0.001

target_language = "spanish" # can be swapped with 'portuguese' as well

fp = open(target_language+"/testset_100000.pkl", 'rb')
testset = cPickle.load(fp)
fp.close()
fp = open(target_language+"/trained_dic_lang1.pkl", 'rb')
dic_lang1 = cPickle.load(fp)
fp.close()
fp = open(target_language+"/trained_dic_lang2.pkl", 'rb')
dic_lang2 = cPickle.load(fp)
fp.close()

fp = open(target_language+"/trained params/trained_we1.pkl", 'rb')
we1 = cPickle.load(fp)
fp.close()
fp = open(target_language+"/trained params/trained_we2.pkl", 'rb')
we2 = cPickle.load(fp)
fp.close()
fp = open(target_language+"/trained params/trained_wx.pkl", 'rb')
wx = cPickle.load(fp)
fp.close()
fp = open(target_language+"/trained params/trained_wh.pkl", 'rb')
wh = cPickle.load(fp)
fp.close()
fp = open(target_language+"/trained params/trained_bh.pkl", 'rb')
bh = cPickle.load(fp)
fp.close()
fp = open(target_language+"/trained params/trained_w.pkl", 'rb')
w = cPickle.load(fp)
fp.close()
fp = open(target_language+"/trained params/trained_b.pkl", 'rb')
b = cPickle.load(fp)
fp.close()
fp = open(target_language+"/trained params/trained_h0.pkl", 'rb')
h0 = cPickle.load(fp)
fp.close()

def contextwin(l, win):
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)
    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]
    assert len(out) == len(l)
    return out

def search_dictionary(dictionary, item):
    if item in dictionary:
        return dictionary[item]
    else:
        return -1

test1_, test2_ = [], []
# User can supply their own 'Target' lanugage data set instead of the provided 'sample' data below.
j = 0
fr = open(target_language+"/english_train_data.txt", 'r')
for line in fr:
    if j in testset:
        test1_.append(line)
    j += 1
fr.close()

j= 0
fr = open(target_language+"/"+target_language+"_train_data.txt", 'r')
for line in fr:
    if j in testset:
        test2_.append(line)
    j += 1
fr.close()

N1 = len(dic_lang1)
N2 = len(dic_lang2)

#########################  Vagueness Detection on English data set  ###########
print "Detecting vagueness in the provided English data set"
rnn = RNN(N1, win, D, H, C, eta, lamb, we1, wx, wh, bh, w, b, h0)
op = open("Outputs/english_op.txt", 'w')
start_time = time.clock()
for i in range(len(test1_)/100):
    lin_ind = map(lambda x: search_dictionary(dic_lang1, x), test1_[i].split())
    cline = contextwin(lin_ind, win)
    y_pred = rnn.test(cline)
    op.write(test1_[i])
    op.write("Vague Words: ")
    for j in range(len(y_pred)):
        if y_pred[j]:
            op.write(test1_[i].split()[j])
            op.write(" ")
    op.write("\n\n")
end_time = time.clock()
op.close()
print "Vagueness Detection took %.2f m" % ((end_time-start_time)/60.0)

#########################  Vagueness Detection on Target language dataset  ####
print "Detecting vagueness in the provided Target language data set"
rnn2 = RNN(N2, win, D, H, C, eta, lamb, we2, wx, wh, bh, w, b, h0)
op = open("Outputs/"+target_language+"_op.txt", 'w')
start_time = time.clock()
for i in range(len(test2_)/100):
    lin_ind = map(lambda x: search_dictionary(dic_lang2, x), test2_[i].split())
    cline = contextwin(lin_ind, win)
    y_pred = rnn2.test(cline)
    op.write(test2_[i])
    op.write("Vague Words: ")
    for j in range(len(y_pred)):
        if y_pred[j]:
            op.write(test2_[i].split()[j])
            op.write(" ")
    op.write("\n\n")
end_time = time.clock()
op.close()
print "Vagueness Detection took %.2f m" % ((end_time-start_time)/60.0)