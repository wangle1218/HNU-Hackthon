# -*- coding:utf-8 -*-

import tensorflow as tf 
import pandas as pd 
import numpy as np 
from utils import *
import os
import datetime
from sklearn.metrics import classification_report, confusion_matrix

class TextCNN(object):
    """
    TextCNN模型
    """
    def __init__(self, sequence_length,vocab,fc_size,num_classes,\
                embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = build_embedding_matrix(vocab,embedding_size)
            self.W = tf.Variable(initial_value = W,name="W",dtype=tf.float32)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size 针对不同大小卷积核的卷积
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # 随机初始化卷积核的权重W
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded, # input
                    W,                            # filter
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            regularizer = tf.contrib.layers.l2_regularizer(l2_reg_lambda)
            fc_layer = tf.layers.dense(self.h_drop, fc_size, activation=tf.nn.elu,\
                                        kernel_regularizer=regularizer, name="fc_layer")
            self.scores = tf.layers.dense(self.h_drop, num_classes, kernel_regularizer=regularizer, name="scores")
            self.predict = tf.argmax(self.scores, 1, name="predict")

        # Calculate loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)
           
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predict, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

class ModelParas(object):
    embedding_dim = 100
    max_document_length = 60
    filter_sizes = "2,3,4,5"
    num_filters = 128
    dropout_keep_prob = 0.5
    l2_reg_lambda = 3.5
    batch_size = 100
    fc_size = 800
    num_epochs = 10
    evaluate_every = 300
    checkpoint_every = 300
    num_checkpoints = 20
    allow_soft_placement = True
    log_device_placement = False


def ModelTrain(data_file, Paras):
    X, y = load_data_and_labels(data_file)
    vocab,vocab_dict = build_vocab(X,3)
    X = build_word2id_matrix(X, vocab_dict, Paras.max_document_length)

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = X[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    shuffle_index = -1 * int(0.1 * float(len(y)))
    x_train, x_dev = x_shuffled[:shuffle_index], x_shuffled[shuffle_index:]
    y_train, y_dev = y_shuffled[:shuffle_index], y_shuffled[shuffle_index:]
    print("Vocabulary Size: {:d}".format(len(vocab)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement = Paras.allow_soft_placement,
          log_device_placement = Paras.log_device_placement)
        sess = tf.Session(config = session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length = x_train.shape[1],
                num_classes = y_train.shape[1],
                vocab = vocab,
                fc_size = Paras.fc_size,
                embedding_size = Paras.embedding_dim,
                filter_sizes = list(map(int, Paras.filter_sizes.split(","))),
                num_filters = Paras.num_filters,
                l2_reg_lambda = Paras.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            model_dir = os.path.abspath(os.path.join(os.path.curdir, "model"))
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=Paras.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: Paras.dropout_keep_prob
                }
                _, step, loss, auc = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy],feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step %30 ==0:
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, auc))

            def dev_step(x_batch, y_batch):
                """
                Evaluates model on a dev set
                """
                predictions = []
                labels = np.argmax(y_batch, 1)

                batches = batch_iter(list(zip(x_batch, y_batch)), 
                                    2048, 
                                    1, 
                                    shuffle=False)
                for batch in batches:
                    x_batchi, y_batchi = zip(*batch)
                    feed_dict = {
                      cnn.input_x: x_batchi,
                      cnn.input_y: y_batchi,
                      cnn.dropout_keep_prob: 1.0
                    }
                    step, loss, accuracy,prediction = sess.run(
                        [global_step, cnn.loss, cnn.accuracy,cnn.predict],feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    prediction = list(prediction)
                    predictions.extend(prediction)

                print("Precision, Recall and F1-Score...")
                print(classification_report(labels, predictions, 
                                            target_names=['neg','pos']))
                print("Confusion Matrix...")
                print(confusion_matrix(labels, predictions))

            # Generate batches
            batches = batch_iter(list(zip(x_train, y_train)), 
                                Paras.batch_size, 
                                Paras.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % Paras.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev)
                    print("")
                if current_step % Paras.checkpoint_every == 0:
                    path = saver.save(sess, './model/model.ckpt', 
                                      global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

def ModelPredict(text, model_path, Paras):
    text = load_data_and_labels(text, train=False)
    _,vocab_dict = build_vocab(text,3)
    text = build_word2id_matrix(text, vocab_dict, Paras.max_document_length)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement = Paras.allow_soft_placement,
          log_device_placement = Paras.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph(model_path + ".meta")
            saver.restore(sess, model_path)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predict").outputs[0]

            # Generate batches for one epoch
            batches = batch_iter(list(text), Paras.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    return all_predictions


class LoadSentiModel(object):
    """docstring for LoadSentiModel"""
    def __init__(self, model_path):
        super(LoadSentiModel, self).__init__()
        self.model_path = model_path
        stop_path = '../data/stopwords_cn.txt'
        self.stop_word = [word.strip() for word in open(stop_path,'r').readlines()]
        dump_path = '../tmp/vocab_dict.pkl'
        self.vocab_dict = pickle.load(open(dump_path,'rb'))

        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement = True,
              log_device_placement = False)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph(self.model_path + ".meta")
                saver.restore(self.sess, self.model_path)

                # Get the placeholders from the graph by name
                self.input_x = graph.get_operation_by_name("input_x").outputs[0]
                self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                self.predictions = graph.get_operation_by_name("output/predict").outputs[0]

    def evaluate(self, text):
        text = jieba.cut(text)
        text = [word for word in text if word not in self.stop_word]
        # print(text)
        mar = [0] * 60
        for i in range(len(text)):
            try:
                mar[i] = self.vocab_dict[text[i]]
            except:
                mar[i] = self.vocab_dict['UNK']

        mar = np.array(mar).reshape(1,-1)
        feed_dict = {self.input_x: mar, 
                    self.dropout_keep_prob: 1.0}
        predictions = self.sess.run(self.predictions, feed_dict)

        return predictions[0]
        

if __name__ == '__main__':
    Paras = ModelParas()
    data_file = '../data/sentiData.csv'
    # ModelTrain(data_file, Paras)

    data_file = '../data/commSongPair_clean.csv'
    df = pd.read_csv(data_file)
    model_path = './model/model.ckpt-2100'
    pred = ModelPredict(data_file, model_path, Paras)
    df['pred'] = pred
    df.to_csv('../data/pred.csv',index=False)
    print(df['pred'].value_counts())

    # model_path = './model/model.ckpt-1200'
    # sModel = LoadSentiModel(model_path)
    # text = '我想要带你去所有的地方，把所有幸福都洒在你脸上'
    # pred = sModel.evaluate(text)
    # print(text,'情感极性:\t',pred)



