# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:48:50 2016

@author: bargav_jayaraman

Description: This script is for training the Vagueness Detection Tool on English followed by cross-lingual knowledge transfer to make the Tool detect vagueness in Target language. This script also evaluates the performance of the Tool using the annotated gold-set (NR-Goldset in the paper).
"""

import numpy
import time
import theano
import cPickle

from RNN import RNN
from AE import AE

win = 7
batch_size = 20
n_epochs = 10
D = 50
C = 2
H = 150
eta = 0.1
lamb = 0.001

target_language = "spanish" # can be swapped with 'portuguese' as well

fp = open(target_language+"/trained_dic_lang1.pkl", 'rb')
dic_lang1 = cPickle.load(fp)
fp.close()
fp = open(target_language+"/trained_dic_lang2.pkl", 'rb')
dic_lang2 = cPickle.load(fp)
fp.close()
fp = open(target_language+"/trainset_100000.pkl", 'rb')
trainset = cPickle.load(fp)
fp.close()

j = 0
train1_, train1_labels = [], []
fr = open(target_language+"/english_train_data.txt", 'r')
for line in fr:
    if j in trainset:
        train1_.append(line)
    j += 1
fr.close()
j = 0
fp = open(target_language+'/english_train_data_labels.txt', 'r')
for line in fp:
    if j in trainset:  
        train1_labels.append(map(int, line.split()))
    j += 1
fp.close()

j= 0
train2_ = []
fr = open(target_language+"/"+target_language+"_train_data.txt", 'r')
for line in fr:
    if j in trainset:
        train2_.append(line)
    j += 1
fr.close()

test1_, test1_labels = [], []
fr = open("annotated dataset/english_annotated.txt", 'r')
for line in fr:
        test1_.append(line)
fr.close()
fr = open("annotated dataset/english_annotated_labels.txt", 'r')
for line in fr:
        test1_labels.append(map(int, line.split()))
fr.close()

test2_, test2_labels = [], []
fr = open("annotated dataset/"+target_language+"_annotated.txt", 'r')
for line in fr:
        test2_.append(line)
fr.close()
fr = open("annotated dataset/"+target_language+"_anotated_labels.txt", 'r')
for line in fr:
        test2_labels.append(map(int, line.split()))
fr.close()

N1 = len(dic_lang1)
N2 = len(dic_lang2)

def contextwin(l, win):
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)
    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]
    assert len(out) == len(l)
    return out

def calc_prec_rec(l1, l2, stp, sfp, sfn, tp, fp, fn):
    fl1, fl2 = 0, 0
    for i in range(len(l1)):
        if l1[i] and l2[i]:
            tp += 1
            fl1 = fl2 = 1
        elif l2[i]:
            fp += 1
            fl2 = 1
        elif l1[i]:
            fn += 1
            fl1 = 1
    if fl1 and fl2:
        stp += 1
    elif fl2:
        sfp += 1
    elif fl1:
        sfn += 1
    return stp, sfp, sfn, tp, fp, fn

def get_prec_rec(tp, fp, fn):
    prec, rec = 1.0, 1.0
    if (tp + fp):
        prec = float(tp)/(tp+fp)
    if (tp + fn):
        rec = float(tp)/(tp+fn)
    return prec, rec

def search_dictionary(dictionary, item):
    if item in dictionary:
        return dictionary[item]
    else:
        return -1

def get_ae_input(i):
    l1 = numpy.zeros(N1+1)
    l2 = numpy.zeros(N2+1)
    for wrd in train1_[i].split():
        l1[search_dictionary(dic_lang1, wrd)] = 1
    for wrd in train2_[i].split():
        l2[search_dictionary(dic_lang2, wrd)] = 1
    return numpy.concatenate([l1, l2])

#########################  Training on English Train Set  #####################    
print "\nTraining the Vagueness Detector on English train set"
rnn = RNN(N1, win, D, H, C, eta, lamb)
avg_cost, prev_cost = 1.0, 2.0
costs = []
epoch = 0
while (epoch < 2*n_epochs) and (avg_cost > 0) and (avg_cost != prev_cost):
    epoch += 1
    prev_cost = avg_cost
    start_time = time.clock()
    for i in range(len(test1_)):
        lin_ind = map(lambda x: search_dictionary(dic_lang1, x), test1_[i].split())
        cline = contextwin(lin_ind, win)
        costs = rnn.train(cline, test1_labels[i])
    end_time = time.clock()
    avg_cost = numpy.mean(costs)
    print "Training of epoch %i took %.2f m" % (epoch, (end_time-start_time)/60.0)
    print 'Average cost for epoch %i is %f' % (epoch, avg_cost)

#########################  Bilingual Training  ################################    
print "\nCross-lingual Knowledge Transfer"
we1 = rnn.get_params()[0]
ae = AE(N1+1, N2+1, D, eta, True, we1)
avg_cost, prev_cost = 1.0, 2.0
costs = []
epoch = 0
while (epoch < n_epochs) and (avg_cost > 0) and (avg_cost != prev_cost):
    epoch += 1
    prev_cost = avg_cost
    start_time = time.clock()
    for i in range(len(train1_)/batch_size):
        train_batch = []
        for j in range(i*batch_size,(i+1)*batch_size):
            train_batch.append(get_ae_input(j))
        costs = ae.train(numpy.array(train_batch).astype(theano.config.floatX))
    end_time = time.clock()
    avg_cost = numpy.mean(costs)
    print "Training of epoch %i took %.2f m" % (epoch, (end_time-start_time)/60.0)
    print 'Average cost for epoch %i is %f' % (epoch, avg_cost)

#########################  Bilingual Training  ################################    
print "\nReverse Knowledge Transfer"
we2 = ae.get_we2()
ae = AE(N1+1, N2+1, D, eta, False, we1, we2)
avg_cost, prev_cost = 1.0, 2.0
costs = []
epoch = 0
while (epoch < n_epochs) and (avg_cost > 0) and (avg_cost != prev_cost):
    epoch += 1
    prev_cost = avg_cost
    start_time = time.clock()
    for i in range(len(train1_)/batch_size):
        train_batch = []
        for j in range(i*batch_size,(i+1)*batch_size):
            train_batch.append(get_ae_input(j))
        costs = ae.train(numpy.array(train_batch).astype(theano.config.floatX))
    end_time = time.clock()
    avg_cost = numpy.mean(costs)
    print "Training of epoch %i took %.2f m" % (epoch, (end_time-start_time)/60.0)
    print 'Average cost for epoch %i is %f' % (epoch, avg_cost)

#########################  Evaluation on English Test Set  ####################
print "\n Evaluating the performance of the Vagueness Detector on English gold-set"
we1 = ae.get_we1()
wx, wh, bh, w, b, h0 = rnn.get_params()[1:]
rnn = RNN(N1, win, D, H, C, eta, lamb, we1, wx, wh, bh, w, b, h0)
stp, sfp, sfn, tp, fp, fn = 0, 0, 0, 0, 0, 0
op = open("english_op.txt", 'w')
start_time = time.clock()
for i in range(len(test1_)):
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
    stp, sfp, sfn, tp, fp, fn = calc_prec_rec(test1_labels[i], y_pred, stp, sfp, sfn, tp, fp, fn)
end_time = time.clock()
op.close()
print "Word-level Evaluation:"
prec, rec = get_prec_rec(tp, fp, fn)
print "Testing took %.2f m" % ((end_time-start_time)/60.0)
print "Average precision is %f" % (prec)
print "Average recall is %f" % (rec)
if (prec + rec):
    print "Average F-score is %f" % (2.0*prec*rec/(prec+rec))
print "Sentence-level Evaluation:"
prec, rec = get_prec_rec(stp, sfp, sfn)
print "Average precision is %f" % (prec)
print "Average recall is %f" % (rec)
if (prec + rec):
    print "Average F-score is %f" % (2.0*prec*rec/(prec+rec))

#########################  Evaluation on Target language Test Set  ############
print "\n Evaluating the performance of the Vagueness Detector on Target language gold-set"
we2 = ae.get_we2()
wx, wh, bh, w, b, h0 = rnn.get_params()[1:]
rnn2 = RNN(N2, win, D, H, C, eta, lamb, we2, wx, wh, bh, w, b, h0)
stp, sfp, sfn, tp, fp, fn = 0, 0, 0, 0, 0, 0
op = open(target_language+"_op.txt", 'w')
start_time = time.clock()
for i in range(len(test2_)):
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
    stp, sfp, sfn, tp, fp, fn = calc_prec_rec(test2_labels[i], y_pred, stp, sfp, sfn, tp, fp, fn)
end_time = time.clock()
op.close()
print "Word-level Evaluation:"
prec, rec = get_prec_rec(tp, fp, fn)
print "Testing took %.2f m" % ((end_time-start_time)/60.0)
print "Average precision is %f" % (prec)
print "Average recall is %f" % (rec)
if (prec + rec):
    print "Average F-score is %f" % (2.0*prec*rec/(prec+rec))
print "Sentence-level Evaluation:"
prec, rec = get_prec_rec(stp, sfp, sfn)
print "Average precision is %f" % (prec)
print "Average recall is %f" % (rec)
if (prec + rec):
    print "Average F-score is %f" % (2.0*prec*rec/(prec+rec))

fp = open(target_language+"/trained params/trained_we1.pkl", 'wb')
cPickle.dump(we1, fp)
fp.close()
fp = open(target_language+"/trained params/trained_we2.pkl", 'wb')
cPickle.dump(we2, fp)
fp.close()
fp = open(target_language+"/trained params/trained_wx.pkl", 'wb')
cPickle.dump(wx, fp)
fp.close()
fp = open(target_language+"/trained params/trained_wh.pkl", 'wb')
cPickle.dump(wh, fp)
fp.close()
fp = open(target_language+"/trained params/trained_bh.pkl", 'wb')
cPickle.dump(bh, fp)
fp.close()
fp = open(target_language+"/trained params/trained_w.pkl", 'wb')
cPickle.dump(w, fp)
fp.close()
fp = open(target_language+"/trained params/trained_b.pkl", 'wb')
cPickle.dump(b, fp)
fp.close()
fp = open(target_language+"/trained params/trained_h0.pkl", 'wb')
cPickle.dump(h0, fp)
fp.close()