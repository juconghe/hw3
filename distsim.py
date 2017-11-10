from __future__ import division
import sys
import json
import math
import os
import numpy as np
from collections import defaultdict


def load_word2vec(filename):
    # Returns a dict containing a {word: numpy array for a dense word vector} mapping.
    # It loads everything into memory.

    w2vec = {}
    with open(filename, "r") as f_in:
        for line in f_in:
            line_split = line.replace("\n", "").split()
            w = line_split[0]
            vec = np.array([float(x) for x in line_split[1:]])
            w2vec[w] = vec
    return w2vec


def load_contexts(filename):
    # Returns a dict containing a {word: contextcount} mapping.
    # It loads everything into memory.

    data = {}
    for word, ccdict in stream_contexts(filename):
        data[word] = ccdict
    print "file %s has contexts for %s words" % (filename, len(data))
    return data


def stream_contexts(filename):
    # Streams through (word, countextcount) pairs.
    # Does NOT load everything at once.
    # This is a Python generator, not a normal function.
    for line in open(filename):
        word, n, ccdict = line.split("\t")
        n = int(n)
        ccdict = json.loads(ccdict)
        yield word, ccdict


def dict_to_vect(d1, d2):
    key1 = set(d1.keys())
    key2 = set(d2.keys())
    vect_1 = np.array([])
    vect_2 = np.array([])
    key1.union(key2)
    for k in key1:
        if k in d1:
            vect_1 = np.append(vect_1, d1[k])
        else:
            vect_1 = np.append(vect_1, 0.0)
        if k in d2:
            vect_2 = np.append(vect_2, d2[k])
        else:
            vect_2 = np.append(vect_2, 0.0)
    return vect_1, vect_2


def cosine(vect_1, vect_2):
    dot_product = vect_1.dot(vect_2)
    size_1 = math.sqrt(np.sum(vect_1 ** 2))
    size_2 = math.sqrt(np.sum(vect_2 ** 2))
    cos = dot_product / (size_1 * size_2)
    return cos


def nearest_word(word_dic, word, nearest_n=20):
    similarity_dic = defaultdict()
    w = word_dic[word]
    for key, value in word_dic.iteritems():
        if not key == word:
            vect_1, vect_2 = dict_to_vect(w, value)
            similarity_dic[key] = cosine(vect_1, vect_2)
    return sorted(similarity_dic.items(), key=lambda (w, c): -c)[:nearest_n]


def nearest_word_dense_vect(vect_dict, word, nearest_n=20):
    similarity_dic = defaultdict()
    word_vect = vect_dict[word]
    for key, vect in vect_dict.iteritems():
        if not key == word:
            similarity_dic[key] = cosine(word_vect, vect)
    return sorted(similarity_dic.items(), key=lambda (w, c): -c)[:nearest_n]


def nearest_word_given_vect(vect_dict, given_vect, nearest_n=20):
    similarity_dic = defaultdict()
    for key, vect in vect_dict.iteritems():
        similarity_dic[key] = cosine(vect, given_vect)
    return sorted(similarity_dic.items(), key=lambda (w, c): -c)[1:nearest_n + 1]
