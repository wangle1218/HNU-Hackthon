# -*- coding:utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
import numpy as np 
import pickle
import os
from utils import *


def sim_pearson(song1,song2):  
    return np.corrcoef(song1, song2)[0][1]

def load_data(data_path):
    df = pd.read_csv(data_path)
    df = df.fillna(' ')
    corpus = []
    for i in range(5000):
        text = df.loc[i,'singer'] + ' '+ df.loc[i,'tfidf_keyWords'] + ' ' + df.loc[i,'lsi_keyWords']
        corpus.append(text)

    tfidf = TfidfVectorizer(ngram_range=(1, 1), max_df=0.8, min_df=3)
    corpus = tfidf.fit_transform(corpus)
    print(corpus.shape)

    return corpus

def trainModel(corpus, model_path):
    if os.path.exists(model_path):
        simMartix = pickle.load(open(model_path,'rb'))
    else:
        cnt = corpus.shape[0]
        simMartix = {}

        for i in range(cnt):
            subMartix = {}
            for j in range(cnt):
                pears = sim_pearson(corpus[i].toarray(),corpus[j].toarray())

                subMartix[j] = np.float16(pears)
            simMartix[i] = subMartix
            # if i % 500 == 0:
            print(i)

        pickle.dump(simMartix,open(model_path,'wb'))

    return simMartix

class LoadRecomModel(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = pickle.load(open(self.model_path,'rb'))

    def recommend(self, songID):
        simSong_dict = self.model[songID]
        simSong_sort = sorted(simSong_dict.items(),key=lambda x:x[1],reverse = True)
        simSong_list = [items[0] for items in simSong_sort[:6]]

        return simSong_list[1:6]



if __name__ == '__main__':
    data_path = '../data/keyWords.csv'
    corpus = load_data(data_path)
    model_path = "../model/recomModel.pkl"
    trainModel(corpus, model_path)

    rModel = LoadRecomModel(model_path)
    a = rModel.recommend(2)
    print(a)







