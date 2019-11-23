# coding: utf-8
import tensorflow as tf
import numpy as np
import json
from sklearn import metrics
from tqdm import tqdm
import os
import re
from bert_layer import *

CONTINUE_TRAIN = False
MAX_SEQLEN = 16
MAX_TEXTLEN = 25
BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 1e-5
ATT_SIZE = 256
MODEL_PATH = 'model/'
DROPOUT_RATE = 0.5
DISPLAY_ITER = 10
BERT_EMBEDDING_SIZE = 768
TRAINSET_PATH = '/bdata/jiayunz/Foursquare/100w/train_1_sentence_emb.json'
TESTSET_PATH = '/bdata/jiayunz/Foursquare/100w/test_1_sentence_emb.json'
BERT_MODEL_PATH = '../../embedding/models/multi_cased_L-12_H-768_A-12'
VENUE_EMBEDDING_PATH = 'venue_embedding.json'
class GenerateData():
    def __init__(self, rpath):
        self.rpath = rpath
        # read data
        self.read_data()
        # shuffle
        self.shuffle()
        # initialize batch id
        self.batch_id = 0

    def read_data(self):
        self.texts = []
        self.venues = []
        self.labels = []
        self.mask = []
        with open(VENUE_EMBEDDING_PATH) as rf:
            venue_embedding = json.load(rf)
        #bc = BertClient()
        with open(self.rpath, 'r') as rf:
            lines = rf.readlines()
            for line in tqdm(lines, total=len(lines)):
                user = json.loads(line.strip())
                #tips = [t['text'] for t in user['fsq']['tips']['tips content'][:MAX_SEQLEN]]
                #emb_tips = bc.encode(tips)
                emb_tips = user['fsq']['tips']['tips embedding'][:MAX_SEQLEN]
                emb_venues = [venue_embedding[t['category']] for t in user['fsq']['tips']['tips content'][:MAX_SEQLEN]]
                self.mask.append([True] * len(emb_tips) + [False] * (MAX_SEQLEN - len(emb_tips)))
                if len(emb_tips) < MAX_SEQLEN:
                    emb_tips = np.concatenate((emb_tips, [[0.] * 768 for _ in range(MAX_SEQLEN - len(emb_tips))]))
                self.texts.append(emb_tips)
                self.venues.append(emb_venues)
                self.labels.append(user['label'])

        self.labels = np.eye(2)[self.labels]

    def next(self):
        if BATCH_SIZE <= len(self.texts) - self.batch_id:
            batch_texts = self.texts[self.batch_id:(self.batch_id + BATCH_SIZE)]
            batch_venues = self.venues[self.batch_id:(self.batch_id + BATCH_SIZE)]
            batch_mask = self.mask[self.batch_id:(self.batch_id + BATCH_SIZE)]
            batch_labels = self.labels[self.batch_id:(self.batch_id + BATCH_SIZE)]
            self.batch_id = self.batch_id + BATCH_SIZE
        else:
            batch_texts = self.texts[self.batch_id:]
            batch_venues = self.venues[self.venues:]
            batch_mask = self.mask[self.batch_id:]
            batch_labels = self.labels[self.batch_id:]

            # shuffle
            self.shuffle()
            # reset batch id
            self.batch_id = 0

        return batch_texts, batch_venues, batch_mask, batch_labels

    def shuffle(self):
        np.random.seed(1117)
        np.random.shuffle(self.texts)
        np.random.seed(1117)
        np.random.shuffle(self.venues)
        np.random.seed(1117)
        np.random.shuffle(self.mask)
        np.random.seed(1117)
        np.random.shuffle(self.labels)

class Model():
    def __init__(self):
        self.build_inputs()
        self.build_model()
        self.build_loss()
        self.build_optimizer()

    def build_inputs(self):
        self.texts = tf.placeholder(shape=[None, MAX_SEQLEN, BERT_EMBEDDING_SIZE], dtype=tf.float32)
        self.venues = tf.placeholder(shape=[None, MAX_SEQLEN, BERT_EMBEDDING_SIZE], dtype=tf.float32)
        self.mask = tf.placeholder(tf.int32, [None, MAX_SEQLEN])
        self.targets = tf.placeholder(tf.float32, (None, 2))
        self.dropout_rate = tf.placeholder(tf.float32)

    def venue_embedding(self, inputs, vocabulary_size, embedding_size):
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embedded = tf.nn.embedding_lookup(embeddings, inputs)
        return embedded

    def attention(self, inputs, mask, return_alphas=False):
        # hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer
        # Trainable parameters
        w_omega = tf.Variable(tf.variance_scaling_initializer(scale=1., mode='fan_in')((int(inputs.shape[-1]), ATT_SIZE)))
        b_omega = tf.Variable(tf.zeros([ATT_SIZE]))
        u_omega = tf.Variable(tf.variance_scaling_initializer(scale=1., mode='fan_in')((ATT_SIZE,)))

        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        # the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1)  # (B,T) shape
        # padding部分减去一个很大的整数，使其softmax后接近于0
        vu = tf.where(tf.equal(mask, True), vu, vu - float('inf'))

        alphas = tf.nn.softmax(vu)  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        if not return_alphas:
            return output
        else:
            return output, alphas

    def build_model(self):
        #embedded_venue = self.venue_embedding(tf.reshape(self.inputs['tip_venue'], (-1,)), 11, config['embedding_size']['tip_venue'])
        #embedded_venue = tf.reshape(embedded_venue, (-1, config['max_seqlen'], config['embedding_size']['tip_venue']))
        hidden_venue, alpha_venue = self.attention(self.venues, self.mask, return_alphas=True)
        hidden_text, alpha_text = self.attention(self.texts, self.mask, return_alphas=True)
        hidden = tf.concat([hidden_text, hidden_venue], axis=1)
        hidden = tf.keras.layers.Dense(units=512, activation='relu')(hidden)
        hidden = tf.keras.layers.Dense(units=128, activation='relu')(hidden)
        hidden = tf.keras.layers.Dense(units=32, activation='relu')(hidden)
        # dropout
        drop_outputs = tf.nn.dropout(hidden, rate=self.dropout_rate)
        self.logits = tf.keras.layers.Dense(units=2, activation=None)(drop_outputs)
        self.probs = tf.nn.softmax(self.logits)
        self.predictions = tf.argmax(self.probs, 1)

    def build_loss(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.targets)
        self.loss = tf.reduce_mean(cross_entropy)

    def build_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.loss)

    def initialize_variables(self, sess):
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        K.set_session(sess)

    def train(self, trainset, testset):
        with tf.Session() as sess:
            if CONTINUE_TRAIN:
                try:
                    saver = tf.train.Saver()
                    saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
                    print('Successfully load model!')
                except:
                    print('Fail to load model!')
                    self.initialize_variables(sess)
            else:
                self.initialize_variables(sess)
            step = 1

            saver = tf.train.Saver()
            while step <= EPOCHS:
                self.train_one_epoch(sess, step, trainset)
                saver.save(sess, MODEL_PATH + 'model')
                print('----- Validation -----')
                self.test(sess, testset)
                print('---------------------')
                step += 1

            print("Optimization Finished!")


    def train_one_epoch(self, sess, step, dataset):
        iter = 0
        while iter * BATCH_SIZE < len(dataset.texts):
            iter += 1
            self.current_epoch = step
            batch_texts, batch_venues, batch_mask, batch_labels = dataset.next()
            feed = {
                self.texts: batch_texts,
                self.venues: batch_venues,
                self.mask: batch_mask,
                self.targets: batch_labels,
                self.dropout_rate: DROPOUT_RATE
            }

            _, y_pred, loss = sess.run([self.optimizer, self.predictions, self.loss], feed_dict=feed)
            y_true = np.argmax(batch_labels, 1)

            if iter % DISPLAY_ITER == 0:
                print("Epoch " + str(step) + ", Iter " + str(iter) +
                      ", Minibatch Loss = " + "{:.5f}".format(loss) +
                      ", Accuracy = " + "{:.5f}".format(metrics.accuracy_score(y_true, y_pred)) +
                      ", f1 score = " + "{:.5f}".format(metrics.f1_score(y_true, y_pred)) +
                      ", AUC = " + "{:.5f}".format(metrics.roc_auc_score(y_true, y_pred))
                      )


    def test(self, sess, dataset):
        #with tf.Session() as sess:
        #    saver = tf.train.Saver()
        #    saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))

        predictions = []
        ground_truth = []

        feed = {
            self.texts: dataset.texts,
            self.venues: dataset.venues,
            self.mask: dataset.mask,
            self.targets: dataset.labels,
            self.dropout_rate: 0.
        }

        y_pred, loss = sess.run([self.predictions, self.loss], feed_dict=feed)
        y_true = np.argmax(dataset.labels, 1)


        predictions.extend(list(y_pred))
        ground_truth.extend(list(y_true))

        print("AUC:", metrics.roc_auc_score(ground_truth, predictions))
        print(metrics.classification_report(ground_truth, predictions, digits=4))


def main(_):
    #start_bert_server()
    trainset = GenerateData(TRAINSET_PATH)
    testset = GenerateData(TESTSET_PATH)
    model = Model()
    #BertServer.shutdown(port=5555)
    model.train(trainset, testset)


if __name__ == '__main__':
    tf.app.run()
