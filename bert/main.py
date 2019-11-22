# coding: utf-8
import tensorflow as tf
import numpy as np
import json
from sklearn import metrics
from tqdm import tqdm
import os
from bert_layer import *

CONTINUE_TRAIN = False
MAX_SEQLEN = 7
MAX_TEXTLEN = 5
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
ATT_SIZE = 64
MODEL_PATH = 'model/'
DROPOUT_RATE = 0.5
DISPLAY_ITER = 1
WORD2VEC_PATH = 'word2vec_model/model'
BERT_HUB_MODULE_HANDLE = 'https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1'
BERT_EMBEDDING_SIZE = 768
N_FINE_TUNE_LAYERS = 3
TRAINSET_PATH = '../../embedding/train_multilingual.json'
TESTSET_PATH = '../../embedding/test_multilingual.json'

if "TFHUB_CACHE_DIR" not in os.environ:
    #os.environ["TFHUB_CACHE_DIR"] = os.path.join("/home/nds/jiayunz/Structure_Hole/embedding/models/", "tfhub")
    os.environ["TFHUB_CACHE_DIR"] = os.path.join("/Users/jiayunz/Study/Structural_Hole/embedding/", "tfhub")


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
        self.text_seq = []
        self.labels = []
        self.mask = []
        with open(self.rpath, 'r') as rf:
            lines = rf.readlines()
            for line in tqdm(lines, total=len(lines)):
                user = json.loads(line.strip())
                tips = [t['text'] for t in user['fsq']['tips']['tips content'][:MAX_SEQLEN]]
                self.mask.append([True] * len(tips) + [False] * (MAX_SEQLEN - len(tips)))
                if len(tips) < MAX_SEQLEN:
                    tips.extend(['None.' for _ in range(MAX_SEQLEN - len(tips))])
                self.text_seq.append(tips)
                self.labels.append(user['label'])

        self.labels = np.eye(2)[self.labels]


    def create_InputExamples(self, data, labels):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, u_tips) in enumerate(data):
            for text in u_tips:
                examples.append(
                    run_classifier.InputExample(
                        guid=None,
                        text_a=text,
                        text_b=None,
                        label=np.argmax(labels[i])
                    )
            )
        return examples

    def next(self):
        if BATCH_SIZE <= len(self.text_seq) - self.batch_id:
            batch_text_seq = self.text_seq[self.batch_id:(self.batch_id + BATCH_SIZE)]
            batch_mask = self.mask[self.batch_id:(self.batch_id + BATCH_SIZE)]
            batch_labels = self.labels[self.batch_id:(self.batch_id + BATCH_SIZE)]
            self.batch_id = self.batch_id + BATCH_SIZE
        else:
            batch_text_seq = self.text_seq[self.batch_id:]
            batch_mask = self.mask[self.batch_id:]
            batch_labels = self.labels[self.batch_id:]

            # shuffle
            self.shuffle()
            # reset batch id
            self.batch_id = 0

        batch_input_examples = self.create_InputExamples(batch_text_seq, batch_labels)

        return batch_input_examples, batch_mask, batch_labels

    def shuffle(self):
        np.random.seed(1117)
        np.random.shuffle(self.text_seq)
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
        self.bert_inputs = dict(
            input_ids=tf.placeholder(shape=[None, MAX_TEXTLEN], dtype=tf.int32),
            input_mask=tf.placeholder(shape=[None, MAX_TEXTLEN], dtype=tf.int32),
            segment_ids=tf.placeholder(shape=[None, MAX_TEXTLEN], dtype=tf.int32)
        )
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
        bert_inputs = [self.bert_inputs['input_ids'], self.bert_inputs['input_mask'], self.bert_inputs['segment_ids']]
        self.emb_text = BertLayer(BERT_HUB_MODULE_HANDLE, n_fine_tune_layers=N_FINE_TUNE_LAYERS)(bert_inputs)
        emb_text = tf.reshape(self.emb_text, (-1, MAX_SEQLEN, BERT_EMBEDDING_SIZE))
        #embedded_venue = self.venue_embedding(tf.reshape(self.inputs['tip_venue'], (-1,)), 11, config['embedding_size']['tip_venue'])
        #embedded_venue = tf.reshape(embedded_venue, (-1, config['max_seqlen'], config['embedding_size']['tip_venue']))
        hidden, alpha = self.attention(emb_text, self.mask, return_alphas=True)

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
                self.run_one_epoch(sess, step, trainset, is_training=True)
                saver.save(sess, MODEL_PATH + 'model')
                print('----- Validation -----')
                self.run_one_epoch(sess, step, testset, is_training=False)
                print('---------------------')
                step += 1

            print("Optimization Finished!")


    def run_one_epoch(self, sess, step, dataset, is_training=False):
        iter = 0
        predictions = []
        ground_truth = []
        while iter * BATCH_SIZE < len(dataset.text_seq):
            iter += 1
            self.current_epoch = step
            batch_text_seq, batch_mask, batch_labels = dataset.next()
            input_ids, input_mask, segment_ids = get_bert_inputs(batch_text_seq, BERT_HUB_MODULE_HANDLE, MAX_TEXTLEN)
            feed = {
                self.bert_inputs['input_ids']: input_ids,
                self.bert_inputs['input_mask']: input_mask,
                self.bert_inputs['segment_ids']: segment_ids,
                self.mask: batch_mask,
                self.targets: batch_labels
            }

            if is_training:
                feed[self.dropout_rate] = DROPOUT_RATE
                sess.run(self.optimizer, feed_dict=feed)
            else:
                feed[self.dropout_rate] = 0.

            y_pred, loss, emb_text = sess.run([self.predictions, self.loss, self.emb_text], feed_dict=feed)
            print(self.emb_text)
            y_true = np.argmax(batch_labels, 1)

            if iter % DISPLAY_ITER == 0:
                print("Epoch " + str(step) + ", Iter " + str(iter) +
                      ", Minibatch Loss = " + "{:.5f}".format(loss) +
                      ", Accuracy = " + "{:.5f}".format(metrics.accuracy_score(y_true, y_pred)) +
                      ", f1 score = " + "{:.5f}".format(metrics.f1_score(y_true, y_pred)) +
                      ", AUC = " + "{:.5f}".format(metrics.roc_auc_score(y_true, y_pred))
                      )

            print(list(y_pred).count(1), list(y_true).count(1))
            predictions.extend(list(y_pred))
            ground_truth.extend(list(y_true))

        print("AUC:", metrics.roc_auc_score(ground_truth, predictions))
        print(metrics.classification_report(ground_truth, predictions, digits=4))


    def test(self, dataset):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
            self.run_one_epoch(sess, 0, dataset, is_training=False)


def main(_):
    trainset = GenerateData(TRAINSET_PATH)
    testset = GenerateData(TESTSET_PATH)
    model = Model()
    model.train(trainset, testset)

if __name__ == '__main__':
    tf.app.run()
