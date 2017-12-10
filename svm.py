#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.svm import LinearSVC, SVC
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

def find_best_c(cs = [10**x for x in range(-10, 10, 1)]):
    train_accs = []
    val_accs = []
    for c in cs:
        model = LinearSVC(loss='hinge', fit_intercept=False, C=c)
        model = model.fit(train_data, train_labels)
        val_acc = model.score(validation_data, validation_labels)
        train_acc = model.score(train_data, train_labels)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
    best_index = val_accs.index(max(val_accs))
    return cs[best_index], cs, train_accs, val_accs


def get_best_model():
    c, cs, train_accs, val_accs = find_best_c()
    model = LinearSVC(loss='hinge', fit_intercept=False, C=c)
    model = model.fit(train_data, train_labels)
    return model


def a(save = False):
    c, cs, train_accs, val_accs = find_best_c()
    axes = plt.gca()
    axes.set_xlim([cs[0], cs[len(cs) -1 ]])
    val, = plt.semilogx(cs, val_accs, marker=(1,0), label='Validation')
    train, = plt.semilogx(cs, train_accs, marker=(4,0), label='Train')
    plt.legend(handles=[val, train])
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Accuracy as a function of C')
    if save:
        plt.savefig('3_a.jpg')
    plt.show()
def c(save = False):
    model = get_best_model()
    w = model.coef_
    w = w.reshape(28, 28)
    plt.imshow(w)
    if save:
        plt.savefig('3_c.jpg')
def d():
    model = get_best_model()
    acc = model.score(test_data, test_labels)
    print 'accuracy of d: ' ,acc
def e():
    model = SVC(gamma=5*1e-7, C=10)
    model = model.fit(train_data, train_labels)
    print 'train accuracy of e ', model.score(train_data, train_labels)
    print 'test accuracy of e ', model.score(test_data, test_labels)
    
            


