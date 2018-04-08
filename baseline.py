import pandas as pd

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import numpy as np
import pickle
import os.path
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

# def featurize(source, title, text, real):
    
#     return {
#         "source": source,
#         "title": title,
#         "text": text,
#         "real": real
#     }

#     return features, real

def main():
    saved_model_file = "baseline_bow.pickle"
    size = 15000

    dataset = pd.read_csv("data/combined.csv")
    all_data = dataset['text'][6000:6000+size].values.astype('U')
    all_labels = dataset['real'].tolist()[6000:6000+size]
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    X_train, X_test, Y_train, Y_true = train_test_split(all_data, all_labels, random_state=42)
    
    if os.path.isfile(saved_model_file):
        print "loading model"
        clf = pickle.load(open(saved_model_file, 'rb'))
    else:
        clf = linear_model.LogisticRegression()
        print "fitting/saving model"
        clf.fit(X_train, Y_train)
        pickle.dump(clf, open(saved_model_file, 'wb'))

    print "featurizing"
    vectorizer = CountVectorizer() 
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    print "predicting"
    Y_pred = clf.predict(X_test)

    results = precision_recall_fscore_support(Y_true, Y_pred, average='macro')
    print results

    print clf.predict(vectorizer.transform(["the earth is flat"]))

if __name__ == '__main__':
    main()