# -*- coding:utf-8 -*-

import pandas as pd 
import numpy as np 
from sklearn import preprocessing
import random
import re
import os
import fasttext
import pickle
import jieba

def split_sent(sent,stop_word):
    sent = re.sub(r'[0-9a-zA-Z]+','',str(sent))
    sent = jieba.cut(sent)
    # print(sent)
    sent = [word for word in sent if word not in stop_word]
    return ' '.join(sent)

def load_data_and_labels(data_file,train=True):
    df = pd.read_csv(data_file,header = 0,encoding='utf8')
    # df_neg = df[df['label']==0]
    # df = pd.concat([df_neg,df],axis=0)
    df = df.sample(frac = 1)

    stop_path = '../data/stopwords_cn.txt'
    stop_word = [word.strip() for word in open(stop_path,'r').readlines()]
    punc_list = "，。、；：“”|~！@#￥%……&*,./?!~;:'（）【】[]_-+=^\'"
    stop_word.extend(list(punc_list))
    stop_word = {word:i for i,word in  enumerate(stop_word)}
    # print(stop_word)

    df["hotComments"] = df["hotComments"].map(lambda x : split_sent(x,stop_word.keys()))
    examples = df["hotComments"].tolist()

    if train==True:
        labels = df["label"].values
        labels = labels.reshape(-1,1)
        ohe = preprocessing.OneHotEncoder()
        labels = ohe.fit_transform(labels).toarray()
        return examples, labels
    else:
        return examples

def build_vocab(text,min_frequency):
    dump_path = '../tmp/vocab.pkl'
    dump_path2 = '../tmp/vocab_dict.pkl'
    if os.path.exists(dump_path):
        vocab = pickle.load(open(dump_path,'rb'))
        vocab_dict = pickle.load(open(dump_path2,'rb'))
    else:
        vocab = dict()
        for line in text:
            for word in line.split():
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        vocab = [keys for keys,value in vocab.items() if value >= min_frequency]
        vocab.append('UNK')
        vocab_dict = {word:i for i,word in enumerate(vocab)}
        pickle.dump(vocab,open(dump_path,'wb'))
        pickle.dump(vocab_dict,open(dump_path2,'wb'))

    return vocab,vocab_dict


def build_word2id_matrix(text,vocab_dict,max_document_length):
    dump_path = '../tmp/word2id_matrix3.pkl'
    if os.path.exists(dump_path):
        word2id_matrix = pickle.load(open(dump_path,'rb'))
    else:
        word2id_matrix = np.zeros((len(text),max_document_length), dtype=int)
        print(word2id_matrix.shape)
        for i in range(len(text)):
            review = text[i].split()
            if len(review) > max_document_length:
                review = review[:max_document_length]
            for j,word in enumerate(review):
                if word in vocab_dict.keys():
                    word2id_matrix[i,j] = vocab_dict[word]
                else:
                    word2id_matrix[i,j] = vocab_dict['UNK']
        # pickle.dump(word2id_matrix,open(dump_path,'wb'))
    return word2id_matrix

def build_embedding_matrix(vocab,embedding_size):
    w2v_model = fasttext.load_model('../w2v/w2v-model.bin')
    embedding_matrix = np.zeros((len(vocab)+1, embedding_size),dtype=float)
    for i in range(1,len(vocab)+1):
        try:
            w2v = w2v_model[vocab[i-1]]
            embedding_matrix[i] = w2v
        except KeyError:
            embedding_matrix[i] = np.random.rand(embedding_size) - 0.5
    del w2v_model

    return embedding_matrix
    
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



