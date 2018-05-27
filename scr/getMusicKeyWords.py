# -*- coding:utf-8 -*-

import pandas as pd 
import numpy as np 
import jieba
import re
import os
import pickle
from gensim import models,corpora,similarities

stop_path = '../data/stopwords_cn.txt'
stop_word = [word.strip() for word in open(stop_path,'r').readlines()]
punc_list = "，。、；：“”|~！@#￥%……&*,./?!~;:'「」（）⋯～╯˙̷⌣҂♪⁍ღ｀ヽ ̛⁎ᵒ＼̴ ̴ ͈ ̶《 》><͈ ̶""【 】[]_-+=^"
stop_word = stop_word + list(punc_list)
# print(stop_word)

def is_num(string):
    try:
        int(string)
        return 1
    except ValueError:
        return 0

def load_clearn_data(data_path):
    df = pd.read_csv(data_path)
    df = df[df['Unnamed: 6'].isnull()]
    df.loc[:,'is_num'] = df['total'].map(lambda x : is_num(x))
    df = df[df['is_num'] == 1]
    df['total'] = df['total'].map(lambda x : int(x))
    df = df[df['total']>999]
    df = df[~df['hotComments'].isnull()]
    df.loc[:,'song_ID'] = list(range(df.shape[0]))
    df['singer'] = df['description'].map(lambda x : x.split('。')[0].split('：')[1])
    df = df[['song_ID','title','singer','url','description','pubDate','hotComments','total']]

    df[['song_ID','title','singer','url','hotComments','total']].to_csv('../data/songFile.csv',index=False)

    return df

def clean_comment(df):
    data = df['hotComments']
    word_dict = {}
    for sent in data:
        sent = re.sub(r'[0-9]+','',str(sent))
        sent = re.sub('\[.*?\]','',sent)
        sent = jieba.cut(sent)
        for word in sent:
            if word not in word_dict.keys():
                word_dict[word] = 1
            else:
                word_dict[word] += 1
    word_list = sorted([w for w in word_dict.keys() if word_dict[w] >= 5])
    word_list = word_list[80 : -643] #+ word_list[-1500:]
    word_list = [w for w in word_list if w not in stop_word]
    word_list = [w for w in word_list if len(w)>1]

    # with open('../data/removeWord.txt','w') as f:
    #     for w in word_list2:
    #         f.write(str(w))
    #         f.write('\n')
    word_dict = {word:i for i,word in enumerate(word_list)}

    def remove(x):
        x = re.sub(r'[0-9]+','',str(x))
        x = jieba.cut(x)
        x = [w for w in x if w in word_dict.keys()]
        return ' '.join(x)

    df['comm_word'] = df['hotComments'].map(lambda x : remove(x))
    df['len'] = df['comm_word'].map(lambda x : len(x.split()))
    df = df[df['len']>=3]
    del df['len']

    return df



def deal_text(text):
    text = text.upper()
    text = re.sub('\[.*?\]','',text)
    text = re.sub(r'[0-9]+','',str(text))
    text = jieba.cut(text, HMM=True)
    text = [word for word in text if len(word)>1]
    text = [word for word in text if word not in stop_word]
    return text

def make_corpus(data):
    corpus = []
    for line in data:
        text = str(line)
        text = deal_text(text)
        corpus.append(text)
    return corpus

def get_tfidf_key_words(df):
    # df = df[:100]
    corpus = make_corpus(df['hotComments'])
    dictionary = corpora.Dictionary(corpus)
    corpus_ = [dictionary.doc2bow(text) for text in corpus]
    tfidf = models.TfidfModel(corpus_)
    corpus_tfidf = tfidf[corpus_]

    key_words_list = []
    for i in range(df.shape[0]):
        doc_tfidf = corpus_tfidf[i]
        sort_doc_tfidf = sorted(doc_tfidf,key = lambda x:x[1],reverse = True)
        key_words = [word_id[0] for word_id in sort_doc_tfidf[:15]]
        key_words = [dictionary[i] for i in key_words]
        key_words = ' '.join(key_words)
        key_words_list.append(key_words)

    df.loc[:,'tfidf_keyWords'] = key_words_list
    df = df[~df['tfidf_keyWords'].isnull()]

    return df

def get_lsi_key_words(df):
    lsi_model_dictionary = '../model/LSI/lsi_dictionary.pkl'
    lsi_model_lsi = '../model/LSI/model.lsi'
    lsi_model_index = '../model/LSI/lsimodel.index'
    if os.path.exists(lsi_model_dictionary):
        dictionary = pickle.load(open(lsi_model_dictionary,'rb'))
        lsi = models.LsiModel.load(lsi_model_lsi)
        index = similarities.MatrixSimilarity.load(lsi_model_index)
    else:
        corpus = []
        for sent in df['comm_word']:
            sent = sent.split()
            corpus.append(sent)

        dictionary = corpora.Dictionary(corpus)
        corpus_ = [dictionary.doc2bow(text) for text in corpus]
        tfidf = models.TfidfModel(corpus_)
        corpus_tfidf = tfidf[corpus_]

        lsi = models.LsiModel(corpus_tfidf, id2word = dictionary, num_topics = 800)
        print("lsi finish!")

        corpus_lsi = lsi[corpus_tfidf]
        index = similarities.MatrixSimilarity(corpus_lsi)
        pickle.dump(dictionary, open(lsi_model_dictionary,'wb'))
        lsi.save(lsi_model_lsi)
        index.save(lsi_model_index)

    topic_list = []
    corpus = []
    for sent in df['comm_word']:
        sent = sent.split()
        corpus.append(sent)
    corpus_ = [dictionary.doc2bow(text) for text in corpus]
    for i,cor in enumerate(corpus_):
        topic_proba = lsi[cor]
        topic_proba = sorted(topic_proba,key = lambda x:x[1],reverse = True)
        try:
            topic_id = topic_proba[0][0]
            topic = lsi.show_topic(topic_id, topn=15)
            topic = [tup[0] for tup in topic if tup[1]>0]
        except:
            topic = []
            print(i,topic)
        topic = ' '.join(topic)
        topic_list.append(topic)

    df.loc[:,'lsi_keyWords'] = topic_list

    return df

def concat_keyWord():
    df1 = pd.read_csv('../data/MusciTFIDFKeyWords.csv')
    df2 = pd.read_csv('../data/MusciLSIKeyWords.csv')
    df2 = df2.drop_duplicates(keep='last')

    song_ID_list = df2['song_ID'].unique()
    key_words = []
    for song_ID in song_ID_list:
        kw = df2.loc[df2['song_ID'] == song_ID,'lsi_keyWords'].tolist()
        kw = ' '.join(kw)
        key_words.append(kw)

    df3 = pd.DataFrame()
    df3['song_ID'] = song_ID_list
    df3['lsi_keyWords'] = key_words

    df1 = pd.merge(df1, df3, how='left', on='song_ID')
    df['song_ID'] = list(range(df.shape[0]))

    df1.to_csv('../data/keyWords.csv',index=False)

    return df1


def get_comm_song_pair(df):
    df = df.reset_index(drop=True)
    song_ID = []
    title_list = []
    comm_list = []

    for i in range(df.shape[0]):
        comment = df.loc[i,'hotComments']
        comment_list = comment.split('|')
        songID = df.loc[i,'song_ID']
        title = df.loc[i,'title']
        songIDs = [songID] * len(comment_list)
        titles = [title] * len(comment_list)

        song_ID.extend(songIDs)
        title_list.extend(titles)
        comm_list.extend(comment_list)

    df_pair = pd.DataFrame()
    df_pair['song_ID'] = song_ID
    df_pair['title'] = title_list
    df_pair['hotComments'] = comm_list

    return df_pair

def remove_highfreq_word(df):
    df['lsi_keyWords'] =df['lsi_keyWords'].astype(str)
    data = df['lsi_keyWords']
    word_dict = {}
    for sent in data:
        sent = str(sent).split()
        for word in sent:
            if word not in word_dict.keys():
                word_dict[word] = 1
            else:
                word_dict[word] += 1
    print(len(word_dict))
    word_list = [word for word in word_dict.keys() if word_dict[word] > 3]

    print(sorted(word_list))
    word_list = [word for word in word_list if word_dict[word] < 1000]

    word_dict = {word:i for i,word in enumerate(word_list)}
    print(len(word_dict))

    df['lsi_keyWords'] = df['lsi_keyWords'].map(lambda x : ' '.join([word for word in x.split() \
                                                                    if word in word_dict.keys()]))
    return df 

if __name__ == '__main__':
    df = load_clearn_data('../music_data/music_reviews2.csv')
    print(df.info())

    df_key = get_tfidf_key_words(df)
    df_key[['song_ID','title','singer','tfidf_keyWords','total']].to_csv('../data/MusciTFIDFKeyWords.csv',index=False)
    print(df_key.info())
    print(df_key.shape)

    df_key = pd.read_csv('../data/MusciKeyWords.csv')
    df_key = remove_highfreq_word(df_key)
    df_key.to_csv('../data/MusciKeyWords2.csv',index=False)

    df_pair = get_comm_song_pair(df)
    df_pair.to_csv('../data/commSongPair.csv',index=False)
    print(df_pair.head())
    print(df_pair.shape)

    df = pd.read_csv('../data/commSongPair.csv')
    print(df.head())
    df = clean_comment(df)
    df.to_csv('../data/commSongPair_clean.csv',index=False)
    print(df.shape)

    df = pd.read_csv('../data/commSongPair_clean.csv')
    df = get_lsi_key_words(df)
    df = df[['song_ID','title','lsi_keyWords']]
    df.to_csv('../data/MusciLSIKeyWords.csv',index=False)

    df = concat_keyWord()
    print(df.shape)



