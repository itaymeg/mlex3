#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import operator
from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean as l2distance

#fetch mnist data
mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

#divide to train and test
idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]
save_folder = os.getcwd()

def our_max(collection, k):
    """
    returns (a,b):
        a - boolean wheter the collection is k sized
        b - if the collection is k sized, return the index to replace
    """
    if len(collection) < k:
        return (False, 0)
    else:
        highest = 0
        for i in xrange(len(collection)):
            if(collection[i][0] > collection[highest][0]): # 0 for distance
                highest = i
        return (True, highest)
def knn(image_set, label_set, query, k):
    distances = []
    for idx, img in enumerate(image_set):
        dist = l2distance(query, img)
        distances.append((idx, dist))
    sortedDistances = sorted(distances, key=operator.itemgetter(1))
    nearest_neightbours = sortedDistances[:k]
    votes = {}
    for idx, (img_idx, dist) in enumerate(nearest_neightbours):
        vote = label_set[img_idx]
        if vote not in votes:
            votes[vote] = 0
        votes[vote] += 1
    sortedVotes = sorted(votes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
    
    
#a
def worse_knn(image_set, label_set, query, k):
    best_k = [] # (distance, label)
    for idx, img in enumerate(image_set):
        dist = l2distance(query, img)
        (k_sized, max_idx) = our_max(best_k, k)
        if not k_sized:
            best_k.append((dist, label_set[idx]))     
        else:
            if dist < best_k[max_idx][0]: # 0 for distance
                best_k[max_idx] = (dist, label_set[idx])
    # find for each label its count in the k best distances
    label_count = {}
    for idx, nn_data in enumerate(best_k):
        dist, label = nn_data
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    # take the label with the most votes
    best_label = None
    for label in label_count:
        if best_label is None:
            best_label = (label, label_count[label])
        else:
            if best_label[1] < label_count[label]:
                best_label = (label, label_count[label])
    return best_label[0]
    
#b
def run_b(k = 10, n = 1000):
    expirment_data = train[:n]
    expirment_labels = train_labels[:n]
    loss_count = 0
    for idx, data in enumerate(test):
        if idx % 250 == 0:
            print "IMG#{0} K#{1} N#{2}".format(str(idx), str(k), str(n))
        prediction = knn(expirment_data, expirment_labels, data, k)
        if prediction != test_labels[idx]:
            loss_count += 1
    accuracy = 100 * (float(len(test)) - float(loss_count)) / float(len(test))
    print "LOSS#{0} TEST#{1} ACC#{2}%".format(str(loss_count), len(test), accuracy)
    return accuracy
        
    
def run_c(save = False):
    print '[EXERCISE C] SAVE={0}'.format(save)
    ks = range(1, 100 + 1)
    stat = []
    for k in ks:
        accuracy = run_b(k=k)
        stat.append(accuracy)
    plt.xlabel('n')
    plt.ylabel('Accuracy')
    plt.plot(ks, stat, 'y')
    if save:
        plt.savefig('figure_C.jpg')
    plt.show()
    
def run_d(save = False):
    print '[EXERCISE D] SAVE={0}'.format(save)
    ns = range(100, 5000 + 1, 100)
    best_k = 1
    stat = []
    for n in ns:
        accuracy = run_b(k=best_k, n=n)
        stat.append(accuracy)
    axes = plt.gca()
    axes.set_xlim([100,5000])
    plt.xlabel('n')
    plt.ylabel('Accuracy')
    plt.plot(ns, stat, 'b')
    if save:
        plt.savefig('figure_D.jpg')
    plt.show()

def main(actions = None, save = False):
    if 'b' in actions:
        run_b()
    if 'c' in actions:
        run_c(save)
    if 'd' in actions:
        run_d(save)
if __name__ == '__main__':
    main(actions= ['c'], save=True)
