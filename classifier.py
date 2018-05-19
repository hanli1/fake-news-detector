import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils import plot_model


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, Embedding, LSTM, SpatialDropout1D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras.preprocessing import text, sequence
from keras import utils


import nltk
from nltk.corpus import stopwords


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def load_more_data():
    midpoint = 13000
    size = 5000
    raw_data = pd.read_csv("data/combined.csv")
    all_data = [cleanText(str(val)) for val in raw_data['text'][int(midpoint-size/2):int(midpoint+size/2)].values]
    all_labels = raw_data['real'][int(midpoint-size/2):int(midpoint+size/2)].values

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)

    x = pd.Series(all_data)
    y = pd.Series(all_labels)

    data = {}
    data['text'] = np.array(list(x))
    data['label'] = np.array(list(y))

    return pd.DataFrame.from_dict(data)

def load_data():
    midpoint = 13000
    size = 10000
    # raw_data2 = pd.read_csv("data/combined.csv")
    raw_data = pd.read_csv("data/fake_or_real_news.csv")

    # all_data = [str(val).replace('\n', '').replace('\t', '') for val in raw_data['text'][int(midpoint-size/2):int(midpoint+size/2)].values]
    all_data = [cleanText(str(val)) for val in raw_data['text'].values]

    # Removes Stop words anFd proper nouns , 
    # stop_words = set(stopwords.words('english'))
    # for i, val in enumerate(all_data):
    #     sentence = str(val)
    #     tagged_sentence = nltk.tag.pos_tag(sentence.split())
    #     edited_sentence = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS' and word != stop_words]
    #     all_data[i] = ' '.join(edited_sentence)

    # all_labels = raw_data['real'][int(midpoint-size/2):int(midpoint+size/2)].values
    all_labels = [0 if val == "FAKE" else 1 for val in raw_data['label'].values]

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)

    x_train, x_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)

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

def cleanText(text):
    # Replace non-ASCII characters with printable ASCII. 
    # Use HTML entities when possible
    if None == text:
        return ''

    
    text = re.sub(r'\x85', '…', text) # replace ellipses
    text = re.sub(r'\x91', "‘", text)  # replace left single quote
    text = re.sub(r'\x92', "’", text)  # replace right single quote
    text = re.sub(r'\x93', '“', text)  # replace left double quote
    text = re.sub(r'\x94', '”', text)  # replace right double quote
    text = re.sub(r'\x95', '•', text)   # replace bullet
    text = re.sub(r'\x96', '-', text)        # replace bullet
    text = re.sub(r'\x99', '™', text)  # replace TM
    text = re.sub(r'\xae', '®', text)    # replace (R)
    text = re.sub(r'\xb0', '°', text)    # replace degree symbol
    text = re.sub(r'\xba', '°', text)    # replace degree symbol

    # Do you want to keep new lines / carriage returns? These are generally 
    # okay and useful for readability
    text = re.sub(r'[\n\r\t]+', ' ', text)     # remove embedded \n and \r

    #removes numbers
    text = re.sub(" \d+", " ", text)

    # This is a hard-core line that strips everything else.
    text = re.sub(r'[\x00-\x1f\x80-\xff]', ' ', text)

    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # tagged_sentence = nltk.tag.pos_tag(text.split())
    # text = ' '.join([word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS' and word != stop_words])
    return text


if __name__ == '__main__':
    start = time.time()
    train_df, test_df = load_data()

    train_text = train_df['text']
    train_label = train_df['label']

    test_text = test_df['text']
    test_label = test_df['label']

    max_words = 1000
    tokenize = text.Tokenizer(num_words=max_words, char_level=False)
    tokenize.fit_on_texts(train_text)

    # print(list(tokenize.word_counts)[:max_words])

    x_train = tokenize.texts_to_matrix(train_text)
    x_test = tokenize.texts_to_matrix(test_text)

    encoder = LabelEncoder()
    encoder.fit(train_label)
    y_train = encoder.transform(train_label)
    y_test = encoder.transform(test_label)

    num_classes = np.max(y_train) + 1
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    print("----- TRAINING DATA INFO -----")
    train_df['label'].value_counts()
    print ("Train size: %d" % len(train_df))
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)

    print("----- TESTING DATA INFO -----")
    test_df['label'].value_counts()
    print ("Test size: %d" % len(test_df))
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)

    batch_size = 32
    epochs = 10

    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))

    # model.add(Embedding(max_words, 128, input_shape=(max_words,)))
    # model.add(SpatialDropout1D(0.4))
    # model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    # model.add(Dense(num_classes))
    # model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

    score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    # plot_model(model, show_shapes=True, to_file='model.png')

    df = load_more_data()

    text = df['text']
    label = df['label']

    x = tokenize.texts_to_matrix(text)
    y = encoder.transform(label)
    y = utils.to_categorical(y, num_classes)

    score = model.evaluate(x, y,
                       batch_size=batch_size, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # y_sigmoid = model.predict(x)
    # y_test_1d = []
    # y_pred_1d = []

    # for i in range(len(y)):
    #     probs = y[i]
    #     index_arr = np.nonzero(probs)
    #     one_hot_index = index_arr[0].item(0)
    #     y_test_1d.append(one_hot_index)

    # for i in range(0, len(y_sigmoid)):
    #     probs = y_sigmoid[i]
    #     predicted_index = np.argmax(probs)
    #     y_pred_1d.append(predicted_index)

    # cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
    # plt.figure(figsize=(12,10))
    # text_labels = encoder.classes_
    # plot_confusion_matrix(cnf_matrix, classes=text_labels, title="Confusion matrix")
    # plt.show()

    print(time.time() - start)