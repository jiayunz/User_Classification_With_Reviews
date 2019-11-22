import gensim
from tqdm import tqdm
import json
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

## 训练自己的词向量，并保存。
def train_word2vec(sentences, size, model_path):
    model = gensim.models.Word2Vec(sentences, size=size, window=5, min_count=1, workers=4) # 训练模型
    model.save(model_path)

def load_tip_text(rpath):
    tips = []
    with open(rpath) as rf:
        lines = rf.readlines()
        for line in tqdm(lines, total=len(lines)):
            u = json.loads(line)
            tips.extend([t['text'] for t in u['fsq']['tips']['tips content']])
    return tips


def load_tip_venue(rpath):
    venue = []
    test = []
    with open(rpath) as rf:
        lines = rf.readlines()
        for line in tqdm(lines, total=len(lines)):
            u = json.loads(line)
            venue.append([t['category'] for t in u['fsq']['tips']['tips content']])
            test.extend([t['category'] for t in u['fsq']['tips']['tips content']])
    return venue


def read_docs(text_list):
    sentences = []
    for i, text in tqdm(enumerate(text_list), total=len(text_list)):
        sentences.append(text_to_word_sequence(text))
    return sentences

def test_word2vec():
    train_tips = ['I like apple.', 'you are a student.', 'I like banana.', 'I like orange', 'She is taller than me.', 'am I from China?']
    test_tips = ['I am a student.', 'I like peach.', 'My sister is from America.']
    x_train = read_docs(train_tips)
    print(x_train)
    model = gensim.models.Word2Vec(x_train, size=10, window=5, min_count=1, workers=4)
    print(model.wv['apple'])


if __name__ == '__main__':
    train_tips = load_tip_text('../embedding/train_en.json')
    X_train = read_docs(train_tips)
    del train_tips
    train_word2vec(X_train, size=200, model_path='word2vec_model/model')