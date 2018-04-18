import pandas as pd
import numpy as np
np.random.seed(42)

from tensorflow.contrib import learn

def get_text_data(midpoint=13000, size=1000, train_split=0.8, vocab_processor=None):
    print("reading data")
    dataset = pd.read_csv("data/combined.csv")
    all_data = dataset['text'][int(midpoint-size/2):int(midpoint+size/2)].values.astype('U')
    all_labels = dataset['real'].tolist()[int(midpoint-size/2):int(midpoint+size/2)]
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    # one hot encoding
    all_labels = np.zeros((all_labels.size, all_labels.max()+1))

    print("train test split")
    shuffle_indices = np.random.permutation(np.arange(len(all_labels)))
    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in all_data])

    if not vocab_processor:
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    
    X = np.array(list(vocab_processor.fit_transform(all_data)))
    Y = all_labels

    X_shuffled = X[shuffle_indices]
    Y_shuffled = Y[shuffle_indices]

    test_sample_index = int(train_split * float(len(Y)))
    X_train, X_test = X_shuffled[:test_sample_index], X_shuffled[test_sample_index:]
    Y_train, Y_true = Y_shuffled[:test_sample_index], Y_shuffled[test_sample_index:]


    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Test split: {:d}/{:d}".format(len(Y_train), len(Y_true)))

    return X_train, Y_train, X_test, Y_true, vocab_processor


def create_batches(data, labels, batch_size, num_epochs):
    for epoch in range(num_epochs):
        # print("Epoch {}/{}".format(epoch, num_epochs))
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        yield data[idx], labels[idx]