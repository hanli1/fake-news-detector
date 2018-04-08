import pandas as pd

from sklearn import linear_model, metrics
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
    saved_model_file = "pickles/baseline_bow.pickle"
    saved_vectorizer = "pickles/baseline_vectorizer.pickle"
    midpoint = 13000
    size = 5000

    print "reading data"
    dataset = pd.read_csv("data/combined.csv")
    all_data = dataset['text'][midpoint-size/2:midpoint+size/2].values.astype('U')
    all_labels = dataset['real'].tolist()[midpoint-size/2:midpoint+size/2]
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)

    print "train test split"
    X_train, X_test, Y_train, Y_true = train_test_split(all_data, all_labels, random_state=42)
    
    if os.path.isfile(saved_model_file):
        print "loading model and vectorizer"
        clf = pickle.load(open(saved_model_file, 'rb'))
        vectorizer = pickle.load(open(saved_vectorizer, 'rb'))
    else:
        print "fitting/saving featurizer"
        vectorizer = TfidfVectorizer(stop_words='english') 
        X_train = vectorizer.fit_transform(X_train)
        pickle.dump(vectorizer, open(saved_vectorizer, 'wb'))

        clf = linear_model.LogisticRegression()
        print "fitting/saving model"
        clf.fit(X_train, Y_train)
        pickle.dump(clf, open(saved_model_file, 'wb'))
    
    X_text = X_test
    X_test = vectorizer.transform(X_test)
    print "predicting"
    Y_pred = clf.predict(X_test)
    results = precision_recall_fscore_support(Y_true, Y_pred, average='macro')
    print results
    # print clf.predict(vectorizer.transform(["barack obama is the president"]))
    print metrics.confusion_matrix(Y_true, Y_pred)
    # for i, (x, prediction, label) in enumerate(zip(X_test, Y_pred, Y_true)):
    #     if prediction != label:
    #         print "predict: {}, actual: {}, \ntext:{}".format(prediction, label, X_text[i]) 

    show_most_informative_features(vectorizer, clf)

def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

if __name__ == '__main__':
    main()