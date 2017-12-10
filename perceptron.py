#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
from sklearn.utils import shuffle
import lib
import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

neg, pos = 0,8
train_idx = np.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
test_idx = np.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

train_data_unscaled = data[train_idx[:6000], :].astype(float)
train_labels = (labels[train_idx[:6000]] == pos)*2-1

validation_data_unscaled = data[train_idx[6000:], :].astype(float)
validation_labels = (labels[train_idx[6000:]] == pos)*2-1

test_data_unscaled = data[60000+test_idx, :].astype(float)
test_labels = (labels[60000+test_idx] == pos)*2-1

# Preprocessing
train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)


train_data = lib.normalize(train_data)
validation_data = lib.normalize(validation_data)
test_data = lib.normalize(test_data)


def perceptron(train, labels):
    """
        train is in shape a,b
        where a is number of training samples
        b in number of features
    """
    w = np.zeros(train.shape[1])
    wrong = True
    while wrong != False:
        wrong = False
        for i, x in enumerate(train):
            prediction = 0
            if w.dot(x) >= 0:
                prediction = 1
            else:
                prediction = -1
            label = labels[i]
            if prediction != label:
                wrong = True
                w += label * x
                
    return w
                

   

def run_a():
    ns = [5, 10, 50, 100, 500, 1000, 5000]
    acc_table = {}
    for n in ns:
        accs = []
        for i in xrange(100):
            if i % 20 == 0:
                print 'N={0}, RUN={1}'.format(n, i)
            s_x, s_y = shuffle(train_data, train_labels)
            h = perceptron(s_x[:n], s_y[:n])
            loss = 0
            for idx,  sample in enumerate(test_data):
                prediction = 0
                if h.dot(sample) >= 0:
                    prediction = 1
                else:
                    prediction = -1
                label = test_labels[idx]
                if prediction != label:
                    loss += 1
            
            acc = (float(test_data.shape[0]) - loss) / float(test_data.shape[0])
            accs.append(acc)
        acc_table[n] = np.mean(accs)
    lib.print_table(acc_table)
def run_b(save = False):
    h = perceptron(train_data, train_labels)
    #print h
    plt.imshow(h.reshape(28, 28), interpolation='nearest')
    if save:
        plt.savefig('2_b.jpg')
    plt.show()
def run_c():
    h = perceptron(train_data, train_labels)
    loss = 0
    for idx,  sample in enumerate(test_data):
        prediction = 0
        if h.dot(sample) >= 0:
            prediction = 1
        else:
            prediction = -1
        label = test_labels[idx]
        if prediction != label:
            loss += 1
            
    acc = (float(test_data.shape[0]) - loss) / float(test_data.shape[0])
    print'accuracy = ', acc
def run_d():
    wrong_samples_idx = []
    h = perceptron(train_data, train_labels)
    loss = 0
    for idx,  sample in enumerate(test_data):
        prediction = 0
        if h.dot(sample) >= 0:
            prediction = 1
        else:
            prediction = -1
        label = test_labels[idx]
        if prediction != label:
            wrong_samples_idx.append(idx)
            loss += 1
    plt.imshow(test_data_unscaled[wrong_samples_idx[1]].reshape(28,28))
    plt.show()
        
if __name__ == '__main__':
    pass
    