# -*- coding:utf-8 -*-

import jieba
import jieba.posseg as pseg
import json
import os
import pickle
import random
import pandas as pd
from sentimentModel import LoadSentiModel
from recommendModel import LoadRecomModel
import warnings
warnings.filterwarnings("ignore")



class Demo(object):
    """
    功能：根据用户的情感状态推荐与其有情感共鸣的歌曲，应用于音乐平台可以提升用户体验和用户留存率。
    1、后端调用这个类运行（代码还未完善，模型加载部分还没写；另外只写了一轮交互，也就是运行这个类后
        只能和用户进行一次交互、给用户推荐一次歌曲）。
    2、对于老用户，如果他本次的情感状态和上次相同或者没有判断出情感状态，则直接推荐和他曾经听过的歌曲相似的歌曲；
        如果不同，就会根据他此时的情感状态来推荐。
    3、代码中需要加载的 csv 或者 txt 文件可以修改为数据库中的表，但是相应的从表中查询数据的代码也需要修改；
        就数据量来说好像不需要用到数据库。
    4、从函数 user_exchange()中进行人机交互，接受用户输入的字符串；交互语句和逻辑可以适当修改。
    5、函数 output() 输出歌曲名称，歌手名字，歌曲热评，歌曲链接（或者像QQ聊天框里的那种歌曲分享网页，可以点击播放），格式为json。
    """
    def __init__(self):
        super(Demo, self).__init__()
        # 加载情感分类模型和推荐模型
        rModel_path = "../model/recomModel.pkl"
        self.rModel = LoadRecomModel(rModel_path)
        model_path = '../model/model.ckpt-2100'
        self.sModel = LoadSentiModel(model_path)
        # 加载情感和歌曲对应关系文件（表头：senti_ID，song_ID，comm_cnt）
        self.senti2song_df = pd.read_csv('../data/senti2song.csv')[:5000]
        # 加载歌曲文件（表头：song_ID，songName，singer，hotComment，songURL）
        self.song_df = pd.read_csv('../data/songFile.csv')[:5000]
        # 加载用户文件（记录用户名字，每行存储一个用户名）
        userFile = '../data/userList.txt'
        if os.path.exists(userFile):
            self.user_list = [user_name.strip() for user_name in open(userFile).readlines()]
        else:
            self.user_list= []
        # 加载老用户听过的歌曲名单（表头：user_name，song_ID，senti_ID）
        user2songFile = '../data/user2songList.csv'
        if os.path.exists(user2songFile):
            self.user2song_df = pd.read_csv(user2songFile)

        # self.text, self.user = self.user_exchange()
        self.user = '赵六'

    def user_exchange(self):
        """
        和用户进行固定交流，判断是新用户还是老用户，将新用户 user_name 存入数据库
        """
        sys.stdout.write("你好，我是会根据你的情绪推荐歌曲的机器人，我可以认识你吗？\n")
        sys.stdout.write("> ")
        sys.stdout.flush()
        input_text = sys.stdin.readline()
        name_flag = True
        user_name = None
        while name_flag:
            input_text = input_text.strip()
            input_cut = pseg.cut(input_text)
            for w in input_cut:
                if w.flag == 'nr':
                    user_name = w.word
                    if user_name not in self.user_list:
                        with open(userFile,'w+') as f:
                            f.write(user_name)
                            f.write('\n')
                    else:
                        self.user_list.append(user_name)

            if user_name is not None:
                name_flag = False
            else:
                sys.stdout.write("你好像还没告诉我你的名字哦~\n")
                sys.stdout.write("> ")
                sys.stdout.flush()
                input_text = sys.stdin.readline()

        if user_name in self.user_list:
            sys.stdout.write(user_name)
            sys.stdout.write("同学，我好想对你有印象，你又来啦~ \n")
        else:
            sys.stdout.write(user_name)
            sys.stdout.write("同学，很高兴认识你~\n")
        sys.stdout.write("> ")
        sys.stdout.flush()
        input_text = sys.stdin.readline()

        return input_text, user_name

    def sentiment_estimate(self, text):
        """
        判断 text 情感倾向
        """
        return self.sModel.evaluate(text)

    def senti2song(self, text):
        """
        根据情感找到与情感类似的歌曲名单
        """
        senti_ID = self.sentiment_estimate(text)
        sort_df = self.senti2song_df.sort_values(by='senti_ID')
        if senti_ID == 1:
            songID_list = sort_df['song_ID'].tolist()[-50:]
        elif senti_ID == 0:
            songID_list = sort_df['song_ID'].tolist()[:50]

        # 取情感类似歌曲的随机五首歌曲
        songID_list = random.sample(songID_list, 5)

        return songID_list

    def simSong_recommend(self, songID):
        """
        如果用户点击了推荐的音乐或者再次登入产品，可以给他推荐和上首他听的歌曲 songID 相似的歌曲
        """
        simSong_list = self.rModel.recommend(songID)

        return simSong_list

    def output(self, songID_list):
        """
        需要输出的
        """
        out_dict = {}
        for i,songID in enumerate(songID_list):
            index = self.song_df[self.song_df['song_ID'] == songID].index
            songName = self.song_df.loc[index, 'title'].tolist()[0]
            singer = self.song_df.loc[index, 'singer'].tolist()[0]
            hotComment = self.song_df.loc[index, 'hotComments'].tolist()[0]
            hotComment = hotComment.split('|')[0]
            songURL = self.song_df.loc[index, 'url'].tolist()[0]

            song_dict = {
                        'songName': songName,
                        'singer': singer,
                        'hotComment': hotComment,
                        'songURL': songURL
                        }

            out_dict[songID] = song_dict

        return json.dumps(out_dict, ensure_ascii=False)

    def senti2RecommSong(self, text):
        """
        返回给后端函数，输出需要按照一定的形式发送给用户
        返回与 text 情感相似的歌曲名称 song_name，歌手名字 singer，歌曲热评 hotComment，歌曲链接 songURL，json格式
        """
        # 判断是否是新用户，新用户按照他输入的字符串进行情感共鸣推荐，老用户根据他之前听过的歌曲推荐
        if self.user == self.user_list[-1]:
            songID_list = self.senti2song(text)
        else:
            senti_ID = self.sentiment_estimate(text)
            pre_senti_ID = self.user2song_df.loc[self.user2song_df['user_name']\
                                                 == self.user,'senti_ID'].tolist()[0]
            
            if senti_ID == pre_senti_ID:
                like_songID = self.user2song_df.loc[self.user2song_df['user_name']\
                                                    == self.user,'song_ID'].tolist()[0]
                songID_list = self.simSong_recommend(like_songID)
            else:
                songID_list = self.senti2song(text)

        out = self.output(songID_list)

        return out

    def recommSimSong(self, like_songID):
        """
        根据用户的喜好歌曲（点击过得歌曲）推荐
        """
        songID_list = self.simSong_recommend(like_songID)

        out = self.output(songID_list)

        return out



if __name__ == '__main__':
    model = Demo()
    S = model.recommSimSong(9)
    print(S)

    text = '我想要带你去所有的地方，把所有幸福都洒在你脸上'
    S = model.senti2RecommSong(text)
    print(S)




