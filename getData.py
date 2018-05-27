import pymongo
import json
import re
import pandas as pd

MONGO_URI = 'mongodb://user:abcd1234@139.198.177.77:2333'
client = pymongo.MongoClient(host="139.198.177.77", port=2333)
db = client["data"]

def getCollection(mdb, name):
    return mdb[name]

hotComments = []
df = pd.DataFrame()

for data in getCollection(db, "shilian").find():
    sent = data['title'] + ' ' + data['content']
    hotComments.append(sent)

df['hotComments'] = hotComments
df['label'] = 0

df['len'] = df['hotComments'].map(lambda x: len(x))
df = df[df['len']>5]
del df['len']


df.to_csv('tieba.csv',index=False)