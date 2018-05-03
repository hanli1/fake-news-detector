import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import time
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords


def load_data():
    midpoint = 13000
    size = 10000
    # raw_data = pd.read_csv("data/combined.csv")
    raw_data = pd.read_csv("data/fake_or_real_news.csv")

    # all_data = [str(val).replace('\n', '').replace('\t', '') for val in raw_data['text'][int(midpoint-size/2):int(midpoint+size/2)].values]
    all_data = [str(val).replace('\n', '').replace('\t', '') for val in raw_data['text'].values]

    #Removes Stop words and proper nouns
    stop_words = set(stopwords.words('english'))
    for i, val in enumerate(all_data):
        sentence = str(val)
        tagged_sentence = nltk.tag.pos_tag(sentence.split())
        edited_sentence = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS' and word != stop_words]
        all_data[i] = ' '.join(edited_sentence)

    # all_labels = raw_data['real'][int(midpoint-size/2):int(midpoint+size/2)].values
    all_labels = [0 if val == "FAKE" else 1 for val in raw_data['label'].values]

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)


    x_train, x_test, y_train, y_test = train_test_split(all_data, all_labels, random_state=42)

    x_train = pd.Series(x_train)
    y_train = pd.Series(y_train)
    x_test = pd.Series(x_test)
    y_test = pd.Series(y_test)

    train = {}
    train['text'] = np.array(list(x_train))
    train['label'] = np.array(list(y_train))

    test = {}
    test['text'] = np.array(list(x_test))
    test['label'] = np.array(list(y_test))

    return pd.DataFrame.from_dict(train), pd.DataFrame.from_dict(test)

if __name__ == '__main__':
    start = time.time()
    train_df, test_df = load_data()

    train_input_fn = tf.estimator.inputs.pandas_input_fn(train_df, train_df['label'], num_epochs=None, shuffle=True)
    # Prediction on the whole training set.
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(train_df, train_df['label'], shuffle=False)
    # Prediction on the test set.
    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(test_df, test_df['label'], shuffle=False)

    embedded_text_feature_column = hub.text_embedding_column(key="text", module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

    estimator = tf.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=2,
        optimizer=tf.train.AdamOptimizer())

    estimator.train(input_fn=train_input_fn, steps=1000);
    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

    print ("Training set accuracy: " + str(train_eval_result))
    print ("Test set accuracy: " + str(test_eval_result))
    print(time.time() - start)

