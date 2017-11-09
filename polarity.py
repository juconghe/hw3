from collections import defaultdict
import math
import numpy as np
import pickle

pos_seed_list = ['good', 'nice', 'love',
                 'excellent', 'fortunate', 'correct', 'superior']
neg_seed_list = ['bad', 'nasty', 'poor',
                 'hate', 'unfortunate', 'wrong', 'inferior']
TOTAL_TWEETS = 174689.0


def tokenize_doc(doc):
    """
    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """
    tokens = doc.split()
    token_set = set()
    for t in tokens:
        if not (t.startswith('@') or t.startswith('#')):
            token_set.add(t.lower())
    return token_set


def construct_dic(doc, word_count_dic, word_pair_count_dic):
    tokens = tokenize_doc(doc)
    is_pos = False
    is_neg = False
    # print tokens
    for token in tokens:
        # print token
        word_count_dic[token] += 1.0
        for p in pos_seed_list:
            if p in tokens:
                word_pair_count_dic[(token, 'POS')] += 1.0
                is_pos = True
                break
        for n in neg_seed_list:
            if n in tokens:
                word_pair_count_dic[(token, 'NEG')] += 1.0
                is_neg = True
                break
    if is_pos:
        word_count_dic['POS'] += 1.0
    if is_neg:
        word_count_dic['NEG'] += 1.0


def calculate_polarity(word_count_dic, word_pair_count_dic, filter=0):
    polarity_dic = defaultdict(float)
    with open('tweets.txt', 'r') as doc:
        for d in doc:
            tokens = tokenize_doc(d)
            for t in tokens:
                if word_count_dic[t] >= filter:
                    if not (t in pos_seed_list or t in neg_seed_list):
                        PMI_pos = 0.0
                        PMI_neg = 0.0
                        p_xy_pos = word_pair_count_dic[(t, 'POS')]
                        p_y_pos = word_count_dic['POS']
                        p_xy_neg = word_pair_count_dic[(t, 'NEG')]
                        p_y_neg = word_count_dic['NEG']
                        if p_xy_pos != 0:
                            PMI_pos = math.log(p_xy_pos / TOTAL_TWEETS) - (math.log(
                                word_count_dic[t] / TOTAL_TWEETS) + math.log(p_y_pos / TOTAL_TWEETS))
                        if p_xy_neg != 0:
                            PMI_neg = math.log(p_xy_neg / TOTAL_TWEETS) - (math.log(
                                word_count_dic[t] / TOTAL_TWEETS) + math.log(p_y_neg / TOTAL_TWEETS))
                        polarity_dic[t] = PMI_pos - PMI_neg
    return polarity_dic


def write_to_pickle(word_count_dic, word_pair_count_dic):
    with open('word_count_dic', 'w') as outfile:
        pickle.dump(word_count_dic, outfile)
    with open('word_pair_count_dic', 'w') as outfile:
        pickle.dump(word_pair_count_dic, outfile)


def load_pickle():
    file1 = open('word_count_dic', 'rb')
    word_count_dic = pickle.load(file1)
    file2 = open('word_pair_count_dic', 'rb')
    word_pair_count_dic = pickle.load(file2)
    return word_count_dic, word_pair_count_dic


if __name__ == '__main__':
    # word_count_dic = defaultdict(float)
    # word_pair_count_dic = defaultdict(float)
    # print 'constructing dictionary'
    # with open('tweets.txt', 'r') as doc:
    #     for d in doc:
    #         construct_dic(d, word_count_dic, word_pair_count_dic)
    #     print 'writing dictionaries to pickle'
    #     write_to_pickle(word_count_dic, word_pair_count_dic)
    word_count_dic, word_pair_count_dic = load_pickle()
    print 'calculating polarity'
    polarity_dic = calculate_polarity(word_count_dic, word_pair_count_dic, 500)
    print '\n=======positive=======\n'
    sorted_polarity = sorted(polarity_dic.items(), key=lambda (w, c): -c)
    for p in sorted_polarity[:50]:
        print p[0], p[1]
    print '\n=======negative=======\n'
    for p in reversed(sorted_polarity[-50:]):
        print p[0], p[1]
