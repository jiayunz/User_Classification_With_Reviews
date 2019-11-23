import tensorflow as tf
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras import backend as K
from tqdm import tqdm
import json
import numpy as np
from sklearn import metrics
import gensim

CONTINUE_TRAIN = False
MAX_SEQLEN = 20
NUM_WORDS = 10000
MAX_TEXTLEN = 64
EMBEDDING_SIZE = 200
FILTERS = 200
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-3
ATT_SIZE = 64
MODEL_PATH = 'model/'
KEEP_PROB = 0.5
DISPLAY_ITER = 1
WORD2VEC_PATH = 'word2vec_model/model'

class GenerateData():
    def __init__(self, rpath, tokenizer=None):
        self.rpath = rpath
        self.tokenizer = tokenizer

        # read data
        self.read_data()
        # preprocessing
        self.preprocessing()
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
                if len(tips) < 1:
                    continue
                self.mask.append([True] * len(tips) + [False] * (MAX_SEQLEN - len(tips)))
                if len(tips) < MAX_SEQLEN:
                    tips.extend(['' for _ in range(MAX_SEQLEN - len(tips))])
                self.text_seq.append(tips)
                self.labels.append(user['label'])

    def preprocessing(self):
        text_reshape = np.reshape(self.text_seq, (-1,))
        # initialize tokenizer
        if not self.tokenizer:
            self.tokenizer = Tokenizer(num_words=NUM_WORDS)
            self.tokenizer.fit_on_texts(text_reshape)
        text_seq = self.tokenizer.texts_to_sequences(text_reshape)
        # padding text
        text_seq_padding = sequence.pad_sequences(text_seq, maxlen=MAX_TEXTLEN)
        self.text_seq = text_seq_padding.reshape((-1, MAX_SEQLEN, MAX_TEXTLEN))

        # get one-hot label
        self.labels = np.eye(2)[self.labels]

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

        return batch_text_seq, batch_mask, batch_labels

    def shuffle(self):
        np.random.seed(1117)
        np.random.shuffle(self.text_seq)
        np.random.seed(1117)
        np.random.shuffle(self.mask)
        np.random.seed(1117)
        np.random.shuffle(self.labels)

def get_word2vec_dictionaries(tokenizer):
    word_index = tokenizer.word_index
    word2vec_model = gensim.models.Word2Vec.load(WORD2VEC_PATH)
    ## 构造包含所有词语的 list，以及初始化 “词语-序号”字典 和 “词向量”矩阵
    vocab_list = [word for word, Vocab in word2vec_model.wv.vocab.items()]  # 存储 所有的 词语

    # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
    # 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如100。
    embeddings_matrix = np.zeros((NUM_WORDS + 1, word2vec_model.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i]  # 每个词语
        if word in word_index and word_index[word] <= NUM_WORDS:
            embeddings_matrix[word_index[word]] = word2vec_model.wv[word]  # 词向量矩阵
    return embeddings_matrix

class TextCNN():
    def __init__(self):
        self.build_inputs()
        self.l2_loss = tf.constant(0.0)
        self.build_model()
        self.build_loss()
        self.build_optimizer()

    def build_inputs(self):
        self.text_seq = tf.placeholder(tf.int32, [None, MAX_SEQLEN, MAX_TEXTLEN])
        self.mask = tf.placeholder(tf.int32, [None, MAX_SEQLEN])
        self.targets = tf.placeholder(tf.float32, (None, 2))
        self.keep_prob = tf.placeholder(tf.float32)

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
        text_seq_reshape = tf.reshape(self.text_seq, (-1, MAX_TEXTLEN))
        #emb_text = tf.keras.layers.Embedding(NUM_WORDS, EMBEDDING_SIZE)(text_seq_reshape)
        emb_text = tf.keras.layers.Embedding(
            NUM_WORDS+1, EMBEDDING_SIZE,
            embeddings_initializer=tf.keras.initializers.constant(embeddings_matrix),
            trainable=True
        )(text_seq_reshape)
        # conv layers
        convs = []
        filter_sizes = [2, 3, 4, 5]
        for fsz in filter_sizes:
            l_conv = tf.keras.layers.Conv1D(filters=FILTERS, kernel_size=fsz, activation='relu')(emb_text)
            l_pool = tf.keras.layers.MaxPooling1D(MAX_TEXTLEN - fsz + 1)(l_conv)
            l_pool_flatten = tf.reshape(l_pool, (-1, FILTERS))
            convs.append(l_pool_flatten)
        merge = tf.keras.layers.concatenate(convs, axis=1)
        merge_seq = tf.reshape(merge, (-1, MAX_SEQLEN, len(filter_sizes) * FILTERS))
        # attention
        hidden, alpha = self.attention(merge_seq, self.mask, return_alphas=True)
        hidden = tf.keras.layers.Dense(units=32, activation='relu')(hidden)
        # dropout
        drop_hidden = tf.nn.dropout(hidden, self.keep_prob)
        self.logits = tf.keras.layers.Dense(units=2, activation=None)(drop_hidden)
        self.probs = tf.nn.softmax(self.logits)
        self.predictions = tf.argmax(self.probs, 1)

    def build_loss(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.targets)
        reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.), tf.trainable_variables())
        self.loss = tf.reduce_mean(cross_entropy + reg)

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
            feed = {
                self.text_seq: batch_text_seq,
                self.mask: batch_mask,
                self.targets: batch_labels
            }

            if is_training:
                feed[self.keep_prob] = KEEP_PROB
                sess.run(self.optimizer, feed_dict=feed)
            else:
                feed[self.keep_prob] = 1.0

            y_pred, loss = sess.run([self.predictions, self.loss], feed_dict=feed)
            y_true = np.argmax(batch_labels, 1)
            if iter % DISPLAY_ITER == 0:
                print(
                    "Epoch " + str(step) + ", Iter " + str(iter) +
                    ", Minibatch Loss = " + "{:.5f}".format(loss) +
                    ", Accuracy = " + "{:.5f}".format(metrics.accuracy_score(y_true, y_pred)) +
                    #", Precision = " + "{:.5f}".format(metrics.precision_score(y_true, y_pred)) +
                    #", Recall = " + "{:.5f}".format(metrics.recall_score(y_true, y_pred)) +
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

if __name__ == '__main__':
    trainset = GenerateData('../../embedding/train.json')
    testset = GenerateData('../../embedding/test.json', trainset.tokenizer)
    embeddings_matrix = get_word2vec_dictionaries(trainset.tokenizer)
    model = TextCNN()
    model.train(trainset, testset)
