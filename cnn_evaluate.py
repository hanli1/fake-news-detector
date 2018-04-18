import numpy as np
np.random.seed(34)
import pickle
import os.path
import sys
import time
import collections
import datetime

from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.contrib import learn

from cnn import TextCNN
import utils


run_num = "1524027820"
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("run_dir", os.path.abspath(os.path.join(os.path.curdir, "runs", run_num)), "run directory from training run")
tf.flags.DEFINE_string("checkpoint_dir", os.path.abspath(os.path.join(os.path.curdir, "runs", run_num, "checkpoints")), "Checkpoint directory from training run")

FLAGS = tf.flags.FLAGS

vocab_path = os.path.join(FLAGS.run_dir, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

X_train, Y_train, X_test, Y_true, _ = utils.get_text_data(vocab_processor=vocab_processor)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph(os.path.join(FLAGS.checkpoint_dir, "model-2.meta"))
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # batches = utils.create_batches(X_test, X_test, FLAGS.batch_size, 1)

        # Collect the predictions here
        batch_predictions = sess.run(predictions, {input_x: X_test, dropout_keep_prob: 1.0})
        all_predictions = np.asarray(batch_predictions)

# convert back into normal 0,1 from one hot encoding
Y_true = np.argmax(Y_true, axis=1)
print(all_predictions.shape)
print(Y_true.shape)

correct_predictions = float(sum(all_predictions == Y_true))
print("Total number of test examples: {}".format(len(Y_true)))
print("Accuracy: {:g}".format(correct_predictions/float(len(Y_true))))