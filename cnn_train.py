import pandas as pd

from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

import tensorflow as tf
from tensorflow.contrib import learn

import numpy as np
import pickle
import os.path
import sys
import time
import collections
from cnn import TextCNN

import datetime

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_float("test_sample_percentage", .2, "Percentage of the training data to use for test")
tf.flags.DEFINE_string("word2vec_file", "data/glove.42B.300d.txt", "Location of Glove pretrained embeddings")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


midpoint = 13000
size = 1000

print("reading data")
dataset = pd.read_csv("data/combined.csv")
all_data = dataset['text'][int(midpoint-size/2):int(midpoint+size/2)].values.astype('U')
all_labels = dataset['real'].tolist()[int(midpoint-size/2):int(midpoint+size/2)]
all_data = np.array(all_data)
all_labels = np.array(all_labels)
# one hot encoding
all_labels = np.zeros((all_labels.size, all_labels.max()+1))

print("train test split")
np.random.seed(42)
shuffle_indices = np.random.permutation(np.arange(len(all_labels)))
# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in all_data])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
X = np.array(list(vocab_processor.fit_transform(all_data)))
Y = all_labels

X_shuffled = X[shuffle_indices]
Y_shuffled = Y[shuffle_indices]

test_sample_index = -1 * int(FLAGS.test_sample_percentage * float(len(Y)))
X_train, X_test = X_shuffled[:test_sample_index], X_shuffled[test_sample_index:]
Y_train, Y_true = Y_shuffled[:test_sample_index], Y_shuffled[test_sample_index:]

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Test split: {:d}/{:d}".format(len(Y_train), len(Y_true)))

# print("fitting/saving")
# clf = Pipeline([ ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
#                  ("logistic regression", linear_model.LogisticRegression())])

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=X_train.shape[1],
            num_classes=2,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters
            )

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))
         
        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
         
        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
         
        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpointing
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        # Tensorflow assumes this directory already exists so we need to create it
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())  


        sess.run(tf.global_variables_initializer())

        # initial matrix with random uniform
        initW = np.random.uniform(-0.25,0.25,(len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
        # # load any vectors from the word2vec
        # with open(FLAGS.word2vec_file, "rb") as lines:
        #     print("Opened word2vec file {}".format(FLAGS.word2vec_file))
        #     w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
        #        for line in lines}

        #     print("word2vec contains " + str(len(w2v)) + " total words")
        #     for word, idx in vocab_processor.vocabulary_._mapping.iteritems():
        #         if word in w2v:
        #             initW[idx] = w2v[word]

        print("Assigning embedding variables")
        sess.run(cnn.W.assign(initW))

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        def create_batches(data, labels, batch_size, num_epochs):
            for epoch in range(num_epochs):
                # print("Epoch {}/{}".format(epoch, num_epochs))
                idx = np.arange(0 , len(data))
                np.random.shuffle(idx)
                idx = idx[:batch_size]
                yield data[idx], labels[idx]

        # Generate batches
        print("generating batches")
        batches = create_batches(X_train, Y_train, FLAGS.batch_size, FLAGS.num_epochs)
        print("training")
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = batch
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(X_test, y_true, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))