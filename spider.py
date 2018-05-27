import requests
import json
import re
import pymongo
import redis
import time
from fc import Fc
from hashlib import sha256
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# ---------- MongoDB Config
MONGO_URI = 'mongodb://data:UVoPR3spBtUtpnYeVhNIG4miTBZl7Gj5@139.198.177.77:2333'
client = pymongo.MongoClient(MONGO_URI)
db = client["music_data"]


def getCollection(mdb, name):
  return mdb[name]


# getCollection(db, "music_info").insert({"123": 1})

# ----------- Redis Cache Config
REDIS_HOST = "139.198.177.77"
REDIS_PORT = 2334

# ----------- Spider Config
exitBuff = True

hostURL = "http://music.163.com"

header = {
  'Host': 'music.163.com',
  'Proxy-Connection': 'keep-alive',
  'Origin': 'http://music.163.com',
  'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36',
  'Content-Type': 'application/x-www-form-urlencoded',
  'Accept': '*/*',
  'Accept-Encoding': 'gzip, deflate',
  'Accept-Language': 'zh-CN,zh;q=0.8',
  'Connection': 'close',
}

nowTasks = 0
maxTimeout = 10
nowTimeout = 0


# ----------- Cache Config

class DoneCache:
  def __init__(self, type):
    self.type = type

  def add(self, data):
    pass

  def check(self, data):
    pass


class PythonSet(DoneCache):
  def __init__(self):
    type = "set"
    super().__init__(type)
    self.__set = set()

  def add(self, data):
    self.__set.add(data)

  def check(self, data):
    return data in self.__set


class RedisCache(DoneCache):
  def __init__(self):
    type = "redis"
    super().__init__(type)
    self.__set = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

  def add(self, data):
    self.__set.sadd('done', data)

  def check(self, data):
    return self.__set.sismember('done', data)


doneSet = RedisCache()


# ------------ Queue Config

class TaskQueue:
  def __init__(self, type):
    self.type = type

  def put(self, data):
    pass

  def get(self):
    pass

  def empty(self):
    pass


class PythonQueue(TaskQueue):
  def __init__(self):
    type = "queue"
    super().__init__(type)
    self.__queue = Queue()

  def put(self, data):
    self.__queue.put(data)

  def get(self):
    try:
      return self.__queue.get_nowait()
    except Exception as e:
      return None

  def empty(self):
    return self.__queue.empty()


class RedisQueue(TaskQueue):
  def __init__(self):
    type = "redis"
    super().__init__(type)
    self.__queue = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

  def put(self, data):
    dealData = str(data[0]) + "," + str(data[1])
    if data[0] == "song":
      self.__queue.lpush("song", dealData)
    else:
      self.__queue.lpush("task", dealData)

  def get(self):
    if not self.empty():
      data = self.__queue.rpop("task").split(",", maxsplit=1)
      if len(data) == 2:
        data = tuple(data)
        return data
    return None

  def getSong(self):
    if not self.emptySong():
      data = self.__queue.rpop("song").split(",", maxsplit=1)
      if len(data) == 2:
        data = tuple(data)
        return data
    return None

  def empty(self):
    return self.__queue.llen("task") == 0

  def emptySong(self):
    return self.__queue.llen("song") == 0


# (type,url)
# ("tag",url)
# ("playlist",url)
# ("song",url)
taskQueue = RedisQueue()


# ------------ Main Function : Crawl Tag

def main():
  mainContent = requests.get(hostURL + '/discover/playlist', headers=header).content.decode("utf-8")
  soup = BeautifulSoup(mainContent, "html.parser")

  for type in soup.find_all(
      lambda x:
      x.has_attr("class") and
      "s-fc1" == x["class"][0] and
      x.name == "a" and
      x.attrs["href"] != None
  ):
    try:
      if len(type.contents) == 1:
        taskQueue.put(("tag", type.attrs["href"]))
    except Exception as e:
      print(e)


# ------------ Tag Function : Crawl PlayList

def getPlayLists(url):
  h = sha256(url.encode("utf-8")).hexdigest()
  if doneSet.check(h):
    print(str(h) + " has been processed")
    return

  try:
    content = requests.get(hostURL + url, headers=header).content.decode("utf-8")
    soup = BeautifulSoup(content, "html.parser")
    for playlist in soup.find_all(
        lambda x:
        x.has_attr("class") and
        "tit" in x["class"] and
        "f-thide" in x["class"] and
        x.name == "a" and
        x.attrs["href"] != None and
        x.attrs["title"] != None
    ):
      try:
        taskQueue.put(("playlist", playlist.attrs["href"]))
        doneSet.add(h)
      except Exception as e:
        print(e)
  except Exception as e:
    print(e)

  global nowTasks
  nowTasks -= 1


# ------------ PlayList Function : Crawl Song

def getSong(url):
  h = sha256(url.encode("utf-8")).hexdigest()
  if doneSet.check(h):
    print(str(h) + " has been processed")
    return

  try:
    content = requests.get(hostURL + url, headers=header).content.decode("utf-8")
    soup = BeautifulSoup(content, "html.parser")
    for song in soup.find_all(
        lambda x:
        x.name == "a" and
        not x.has_attr("class") and
        x.has_attr("href")
    ):
      try:
        songList = re.findall("/song\?id=(\d+)", song.attrs["href"])
        if len(songList) > 0:
          taskQueue.put(("song", int(songList[0])))
          doneSet.add(h)
      except Exception as e:
        print(e)
  except Exception as e:
    print(e)

  global nowTasks
  nowTasks -= 1


# ------------ Song Function : Crawl Information of Song

def getHotComments(id):
  if doneSet.check(id):
    print(str(id) + " has been processed")
    return

  commentAPI = hostURL + '/weapi/v1/resource/comments/R_SO_4_' + str(id) + '?csrf_token='  # 歌评 API
  lyricAPI = hostURL + '/weapi/song/lyric?csrf_token=' + str(id) + '?csrf_token='  # 歌词 API

  commentData = {
    'params': '7gmBqqmv5yU97gXmkaXnrMTSp4m3Y9UwWcEsAUAuUTvIGCtbUcmVdmETzSeHYMuVg/6+pYn78DRYsGw1yn2im2VIg9R1Y0n8c+Q8rMfhm07uDh3EQlgXR2xiQ4VwNqfcD4IrxY9peRihzhaAGiy2Amq6B7eC5DHqhNhGDeSnToXNt8SaSviDCVcnX+silmp6',
    'encSecKey': 'c470ceb49c0bcfa81cd7748d8a893d7c7baf2a7749ba64714a089587e7eddbe7d360fd46efaf044747f04b3fc0f3b68fdef5d77a297f53ee369700aa45eaadcd920983307c7517df4f44f16de6f22bdf610d389916658b1444a92f0cf56c975d15c4245e3f506004fe08c0f6a093560f35a642d04fb9b055f39a8a4c71391a54'
  }
  lyricData = {
    'params': '1lxKyLsIRBOVhN7fbbuPeoCMmMD8W+DxM14F5kKsYKIYs+ylxS7Dp5G1Gmsd5aHcQnU1xF7M5ISLNVOTw0V8S3oBMO4ZkYnXHjfjD6zRr8NJjzQei8L5A0jfJhLf/wWx',
    'encSecKey': '3ef98f1385a396fc19d4623d6ee7e778998ce28795e749b782ca6dad4a630a5cc00cf7af9814e5a4ff1161c434c872d145272927fb997d8d5086feddeae5c666faeb89803bb1beda13b8149a4353a850b5bd3610fc521a9c0f4b7c503bd494a3c6db83fbb9d193512a5bc26627fcb8467cd80bae6192cc132f68dfcbabb56691'
  }
  try:
    htmlContent = requests.get(hostURL + "/song?id=" + str(id), headers=header).content.decode("utf-8")
    soup = BeautifulSoup(htmlContent, "html.parser")
    tag = soup.find(
      lambda x:
      x.has_attr("type") and
      "application/ld+json" == x["type"] and
      x.name == "script"
    )

    songInfoObj = json.loads(tag.string)
    commentsObject = json.loads(requests.post(commentAPI, data=commentData, headers=header).content.decode("utf-8"))
    lyricObject = json.loads(requests.post(lyricAPI, data=lyricData, headers=header).content.decode("utf-8"))

    saveObj = {
      "songInfoObj": songInfoObj,
      "lyricObject": lyricObject,
      "commentsObject": commentsObject,
    }

    getCollection(db, "music_info").insert(saveObj)

    doneSet.add(id)
  except Exception as e:
    taskQueue.put(("song", id))
    print(e)

  global nowTasks
  nowTasks -= 1
  # if taskQueue.empty():
  #   global exitBuff
  #   exitBuff = False
  #   print("Now,the taskQueue is Empty.\nYou can send Ctrl+C Sign to exit.")


if __name__ == "__main__":
  # main()
  # taskQueue.put(("playlist", '/playlist?id=2195453350'))
  # taskQueue.put(("song", 534066094))
  pool = ThreadPoolExecutor(200)

  # Make Tasks
  # while not taskQueue.empty() or nowTasks > 0:
  #   try:
  #     taskInfo = taskQueue.get()
  #     if taskInfo == None:
  #       waitTime = 0.5
  #       print("Empty Task,exit after {t} seconds".format(t=maxTimeout - nowTimeout))
  #       if nowTimeout >= maxTimeout:
  #         break
  #       nowTimeout += waitTime
  #       time.sleep(waitTime)
  #       continue
  #     if taskInfo[0] == "tag":
  #       print("Deal with tag:{url}".format(url=taskInfo[1]))
  #       pool.submit(getPlayLists, taskInfo[1])
  #     elif taskInfo[0] == "playlist":
  #       print("Deal with playlist:{url}".format(url=taskInfo[1]))
  #       pool.submit(getSong, taskInfo[1])
  #     elif taskInfo[0] == "song":
  #       print("Deal with song:{id}".format(id=str(taskInfo[1])))
  #       pool.submit(getHotComments, taskInfo[1])
  #     nowTimeout = 0
  #   except Exception as e:
  #     print(e)

  # Consumer Tasks
  while not taskQueue.emptySong() or nowTasks > 0:
    try:
      taskInfo = taskQueue.getSong()
      if taskInfo == None:
        waitTime = 0.5
        print("Empty Task,exit after {t} seconds".format(t=maxTimeout - nowTimeout))
        if nowTimeout >= maxTimeout:
          break
        nowTimeout += waitTime
        time.sleep(waitTime)
        continue

      if taskInfo[0] == "song":
        print("Deal with song:{id}".format(id=str(taskInfo[1])))
        pool.submit(getHotComments, taskInfo[1])
      nowTimeout = 0
    except Exception as e:
      print(e)

  pool.shutdown()
  print("exit successfully!")
