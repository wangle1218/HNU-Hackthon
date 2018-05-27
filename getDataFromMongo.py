import pymongo
import json
import re
import pandas as pd

MONGO_URI = 'mongodb://data:UVoPR3spBtUtpnYeVhNIG4miTBZl7Gj5@139.198.177.77:2333'
client = pymongo.MongoClient(MONGO_URI)
db = client["music_data"]


def getCollection(mdb, name):
    return mdb[name]

def parse(data):
    song_title = data['songInfoObj']['title']
    des = data['songInfoObj']['description']
    song_url = data['songInfoObj']['@id']
    pubDate = data['songInfoObj']['pubDate']
    hotComments = []
    hc = data['commentsObject']['hotComments']
    for comm in hc:
        hotComments.append(comm['content'])
    hotComments = '|'.join(hotComments)
    hotComments = re.sub(',','，',hotComments)
    comm_total = data['commentsObject']['total']

    return [song_title,song_url,des,str(pubDate),hotComments,str(comm_total)]

def deal_csv(data_file):
    datas = []
    pre_line = ''
    with open(data_file,'r') as f:
        for line in f.readlines():
            if 'http://' not in line:
                pre_line = pre_line +'。'+ line.strip()
            else:
                pre_line = pre_line.split(',')
                datas.append(pre_line)
                pre_line = line.strip()

    datas = pd.DataFrame(datas)
    datas.to_csv('./music_data/music_reviews2.csv',index=False)


if __name__ == "__main__":
    # i = 0
    # fo = open('./music_data/music_reviews.csv', 'w')
    # data = ['title','url','description','pubDate','hotComments','total']
    # data = ','.join(data)
    # fo.write(data + '\n')
    # for data in getCollection(db, "music_info").find():
    #     # print(data.keys())
    #     data = parse(data)
    #     # datas.append(data)
    #     data = ','.join(data) + '\n'
    #     fo.write(data)

    #     i += 1
    #     if i % 10000 ==0:
    #         print(data)

    # fo.close()
    deal_csv('./music_data/music_reviews.txt')

