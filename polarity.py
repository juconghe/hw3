from collections import defaultdict
import math
import numpy as np
import pickle

polarity_dic = defaultdict(float)
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
    lowered_tokens = [t.lower() for t in tokens if not (
        t.startswith('@') or t.startswith('#'))]
    return set(lowered_tokens)


def construct_dic(doc, word_count_dic, word_pair_count_dic):
    tokens = tokenize_doc(doc)
    # print tokens
    for token in tokens:
        # print token
        word_count_dic[token] += 1.0
        for p in pos_seed_list:
            if p in tokens:
                word_pair_count_dic[(token, p)] += 1.0
        for n in neg_seed_list:
            if n in tokens:
                word_pair_count_dic[(token, n)] += 1.0


def calculate_polarity(word_count_dic, word_pair_count_dic):
    with open('tweets.txt', 'r') as doc:
        for d in doc:
            tokens = tokenize_doc(d)
            for t in tokens:
                p_wy_pos = []
                p_wy_neg = []
                if t not in pos_seed_list:
                    for p in pos_seed_list:
                        p_xy = 0 if word_pair_count_dic[t, p] == 0 else math.log10(
                            word_pair_count_dic[t, p] / TOTAL_TWEETS)
                        p_x = math.log10(word_count_dic[t] / TOTAL_TWEETS)
                        p_y = math.log10(word_count_dic[p] / TOTAL_TWEETS)
                        PMI = p_xy - (p_x + p_y)
                        p_wy_pos.append(PMI)
                if t not in neg_seed_list:
                    for n in neg_seed_list:
                        p_xy = 0 if word_pair_count_dic[t, n] == 0 else math.log10(
                            word_pair_count_dic[t, n] / TOTAL_TWEETS)
                        p_x = math.log10(word_count_dic[t] / TOTAL_TWEETS)
                        p_y = math.log10(word_count_dic[n] / TOTAL_TWEETS)
                        PMI = p_xy - (p_x + p_y)
                        p_wy_neg.append(PMI)
                polarity_dic[t] = np.mean(p_wy_pos) - np.mean(p_wy_neg)


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
    calculate_polarity(word_count_dic, word_pair_count_dic)
    print 'positive'
    for p in sorted(polarity_dic.items(), key=lambda (w, c): -c)[:50]:
        print p
    print 'negative'
    for p in sorted(polarity_dic.items(), key=lambda (w, c): c)[:50]:
        print p
