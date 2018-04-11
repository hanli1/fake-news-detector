import pandas as pd

from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

import numpy as np
import pickle
import os.path
import sys
import time
import collections
reload(sys)
sys.setdefaultencoding("utf-8")

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = collections.defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

def main():
    midpoint = 13000
    size = 1000

    print "reading data"
    dataset = pd.read_csv("data/combined.csv")
    all_data = dataset['text'][midpoint-size/2:midpoint+size/2].values.astype('U')
    all_data = [word_tokenize(sentence) for sentence in all_data]
    all_labels = dataset['real'].tolist()[midpoint-size/2:midpoint+size/2]
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)

    # model = Word2Vec(all_data, size=100)
    # w2v = dict(zip(model.wv.index2word, model.wv.syn0))

    #Download glove file from https://github.com/stanfordnlp/GloVe
    with open("data/glove.42B.300d.txt", "rb") as lines:
        w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
               for line in lines}

    print "train test split"
    X_train, X_test, Y_train, Y_true = train_test_split(all_data, all_labels, random_state=42)
    
    print "fitting/saving featurizer"
    clf = Pipeline([ ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                     ("logistic regression", linear_model.LogisticRegression())])

    print "fitting/saving model"
    clf.fit(X_train, Y_train)
    
    X_text = X_test
    print "predicting"
    Y_pred = clf.predict(X_test)
    results = precision_recall_fscore_support(Y_true, Y_pred, average='macro')
    print results
    # print clf.predict(vectorizer.transform(["barack obama is the president"]))
    print metrics.confusion_matrix(Y_true, Y_pred)


if __name__ == '__main__':
    start = time.time()
    main()
    print (time.time() - start)