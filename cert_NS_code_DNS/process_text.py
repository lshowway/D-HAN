'''
没有实体的时候，使用all的向量
'''
import bcolz
import numpy as np
import codecs
import re
import jieba
import os, json
from gensim.models.doc2vec import Doc2Vec

def loadStopWords():
    stopwords = set([])
    with codecs.open("./data/stopwords.txt", "r", "utf-8") as simpleFile:
        for line in simpleFile:
            stopwords.add(line.strip())
    return stopwords

def split_sentence(text, punctuation_list='!?。！？'):
    sentence_set = []
    inx_position = 0  # 索引标点符号的位置
    char_position = 0  # 移动字符指针位置
    for char in text:
        char_position += 1
        if char in punctuation_list:
            next_char = list(text[inx_position:char_position + 1]).pop()
            if next_char not in punctuation_list:
                sentence_set.append(text[inx_position:char_position])
                inx_position = char_position
    if inx_position < len(text):
        sentence_set.append(text[inx_position:])

    # sentence_with_index = {i: sent for i, sent in enumerate(sentence_set)}
    return sentence_set


def SentenceNumber():
    with codecs.open("./caixin/data(L=10)/newsid_tc(L=10).txt", "r", "utf-8") as f:
        line = f.readline()
        count = 0
        count1 = 0
        count5 = 0
        lenList = []
        while line:

            #第二行
            line = f.readline()

            # 第三行
            line = f.readline()
            temp = split_sentence(line)
            lenList.append(len(temp))
            if len(temp)==1:
                count1 += 1
            if len(temp)<=20:
                count5 += 1

            #第四行
            line = f.readline()

            # 第一行
            line = f.readline()

            count += 1

        lenList.sort()
        print("文件大小：" + str(count))
        print("len(lenList):" + str(len(lenList)))
        print("int(len(lenList)*0.8):" + str(int(len(lenList) * 0.8)))
        print("80%分界点:" + str(lenList[int(len(lenList) * 0.8)]))  # 80%分界点:32
        print("80%分界点2:" + str(lenList[int(len(lenList) * 0.8) - 1]))  # 80%分界点2:32
        print(count1)
        print(count5)


def process1():
    newsSet = set([])
    with codecs.open("./caixin/nocontent78.txt", "r", "utf-8") as nocontent:
        for line in nocontent:
            newsSet.add(line.strip())
    print("没有内容的新闻：" + str(len(newsSet)))

    count = 0
    count2 = 0
    with open('./caixin/train_2.txt', 'w', encoding='UTF-8') as f:
        with codecs.open("./caixin/train.txt", "r", "utf-8") as trainFile:
            for line in trainFile:
                if (ord(line[0]) == 65279):
                    line = line[1:]

                result = line.strip().split("\t")
                if (len(result) != 6):
                    print("该行属性不为6！")

                if result[1] in newsSet:
                    count2 += 1
                    continue
                else:
                    f.write(line.strip() + "\n")
                    count += 1
    print("train_2.txt size:" + str(count)) # 113360
    print("delete size:" + str(count2)) # 2865


def readNumber():
    count = 0
    userReadNum = {}
    with open('./caixin/readNumber.txt', 'w', encoding='UTF-8') as f:
        with codecs.open("./caixin/train_2.txt", "r", "utf-8") as trainFile:
            for line in trainFile:
                if (ord(line[0]) == 65279):
                    line = line[1:]

                result = line.strip().split("\t")
                if (len(result) != 6):
                    print("该行属性不为6！")

                uid = result[0].strip()
                if uid in userReadNum:
                    temp = userReadNum[uid]
                    userReadNum[uid] = temp + 1
                else:
                    userReadNum[uid] = 1
                count += 1

            for key, value in userReadNum.items():
                f.write(str(key) + ":" + str(value) + "\n")
    print(len(userReadNum))  # 10000
    print(count)  # 113360


def process2():
    # 读用户阅读的新闻数目
    userSet = set([]) # int
    with codecs.open("./caixin/readNumber.txt", "r", "utf-8") as readNumberFile:
        for line in readNumberFile:
            result = line.strip().split(":")
            if int(result[1])<12:
                userSet.add(result[0])
    print("userSet size:"+str(len(userSet))) # 8053

    count = 0
    count2 = 0
    with open('./caixin/train_3(L=10).txt', 'w', encoding='UTF-8') as f:
        with codecs.open("./caixin/train_2.txt", "r", "utf-8") as trainFile:
            for line in trainFile:
                if (ord(line[0]) == 65279):
                    line = line[1:]

                result = line.strip().split("\t")
                if(len(result)!=6):
                    print("该行属性不为6！")

                if result[0] in userSet:
                    count2 += 1
                    continue
                else:
                    f.write(line.strip()+"\n")
                    count += 1
    print("train_3(L=10).txt size:" + str(count)) # 61615
    print("delete size:" + str(count2)) # 51745


def process3():
    lastUid = ""
    lastReadTime = ""
    count = 0
    count_train = 0
    count_test= 0
    newsSet = set([])
    with open('./caixin/newsid_tc(L=10).txt', 'w', encoding='UTF-8') as tc:
        with open('./caixin/userSeq_train(L=10).txt', 'w', encoding='UTF-8') as train:
            with open('./caixin/userSeq_test(L=10).txt', 'w', encoding='UTF-8') as test:
                with codecs.open("./caixin/train_3(L=10).txt", "r", "utf-8") as trainFile:
                    for line in trainFile:
                        result = line.strip().split("\t")
                        uid = result[0].strip()
                        newsid = result[1].strip()
                        readTime = result[2].strip()
                        if newsid not in newsSet:
                            tc.write("id:"+newsid+"\n")
                            tc.write("title:"+ result[3] +"\n")
                            tc.write("content:" + result[4] +"\n")
                            tc.write("pubTime:"+ result[5] +"\n")
                            newsSet.add(newsid)

                        if count == 0 :
                            test.write(uid + "\t" + newsid + "\t" + readTime + "\n")

                            lastUid = uid
                            lastReadTime = readTime
                            count_test += 1
                        else:
                            if uid !=lastUid: # test
                                test.write(uid + "\t" + newsid + "\t" + readTime + "\n")

                                lastUid = uid
                                lastReadTime = readTime
                                if (lastReadTime < readTime):
                                    print("其中有不按顺序的。")
                                count_test += 1
                            else :# train
                                train.write(uid + "\t" + newsid + "\t" + readTime + "\n")
                                lastUid = uid
                                lastReadTime = readTime
                                if(lastReadTime < readTime):
                                    print("其中有不按顺序的。")
                                count_train += 1

                        count += 1
    print(count)  # 61615
    print(count_train) # 59668
    print(count_test) # 1947
    print(len(newsSet)) # 5275

# 得到newsid_sen_word20
def process4():
    # 加载新闻内容并进行处理,得到文件:
    # news编号(int)
    # word1,word2,word3,|| word4,word5,|| ...

    # 加载停止词
    stoplist = loadStopWords()
    print(len(stoplist))

    # 加载词向量
    embed = {}
    folder_path = './data/zh.64/zh.64/'
    words, embeddings = load_embeddings(folder_path)  # embeddings中的数据类型是numpy.float32
    for i, w in enumerate(words):
        embed[w] = embeddings[i]
    print(len(embed))

    # 读文件， 写文件
    with open('./caixin/newsid_sen_word20(L=10).txt', 'w', encoding='UTF-8') as f:
        pattern = re.compile(u'[\u4e00-\u9fa5]+')
        article_in = open('./caixin/newsid_tc(L=10).txt', 'r', encoding='UTF-8')
        line = article_in.readline()
        count = 0
        while line:
            id = line[3:].strip()

            # 第二行：title
            line = article_in.readline()
            title = line[6:].strip()
            # 第三行：content
            line = article_in.readline()
            content = line[8:].strip()

            tc = title + "。" + content

            news = []  # 存放 sentence 的 list
            temp = split_sentence(tc)
            for s in temp:  # 循环每个句子,去停止词,去掉字母，数字，标点,去掉没有embedding的词
                sentence = jieba.cut(s, cut_all=False)
                sentence = [word for word in sentence if
                            word not in stoplist and pattern.search(word) and word in embed]
                if len(sentence) == 0:  # 如果是一个空句子，这句话不加入news
                    continue
                news.append(sentence)
            if (len(news) > 20):  # 每条新闻最多处理20个句子
                news = news[0:20]
            if (len(news) == 0):
                print("有不含单词的新闻" + str(id))

            f.write(id + "\n")
            for sen in news:
                for word in sen:
                    f.write(word + ",")
                f.write("||")
            f.write("\n")

            # 第四行：pubTime ，第一行：id
            line = article_in.readline()
            line = article_in.readline()

            count += 1
            if count % 10000 == 0:
                print(count)
    article_in.close()

# 得到newsid_word
def process5():
    # 加载新闻内容并进行处理,得到文件:
    # news编号(int)
    # word1,word2,word3,|| word4,word5,|| ...

    # 加载停止词
    stoplist = loadStopWords()
    print(len(stoplist))

    # 加载词向量
    embed = {}
    folder_path = './data/zh.64/zh.64/'
    words, embeddings = load_embeddings(folder_path)  # embeddings中的数据类型是numpy.float32
    for i, w in enumerate(words):
        embed[w] = embeddings[i]
    print(len(embed))

    # 读文件， 写文件
    with open('./caixin/newsid_word(L=10).txt', 'w', encoding='UTF-8') as f:
        pattern = re.compile(u'[\u4e00-\u9fa5]+')
        article_in = open('./caixin/newsid_tc(L=10).txt', 'r', encoding='UTF-8')
        line = article_in.readline()
        count = 0
        while line:
            id = line[3:].strip()

            # 第二行：title
            line = article_in.readline()
            title = line[6:].strip()
            # 第三行：content
            line = article_in.readline()
            content = line[8:].strip()

            tc = title + "。" + content

            news = []  # 存放 sentence 的 list
            temp = split_sentence(tc)
            for s in temp:  # 循环每个句子,去停止词,去掉字母，数字，标点,去掉没有embedding的词
                sentence = jieba.cut(s, cut_all=False)
                sentence = [word for word in sentence if
                            word not in stoplist and pattern.search(word) and word in embed]
                if len(sentence) == 0:  # 如果是一个空句子，这句话不加入news
                    continue
                news.append(sentence)
            if (len(news) == 0):
                print("有不含单词的新闻" + str(id))

            f.write(id + "\n")
            for sen in news:
                for word in sen:
                    f.write(word + ",")
                f.write("||")
            f.write("\n")

            # 第四行：pubTime ，第一行：id
            line = article_in.readline()
            line = article_in.readline()

            count += 1
            if count % 10000 == 0:
                print(count)
    article_in.close()

def load_embeddings(folder_path):
    """从 bcolz 加载 词/字 向量
    Args:
        - folder_path (str): 解压后的 bcolz rootdir（如 zh.64），
                             里面包含 2 个子目录 embeddings 和 words，
                             分别存储 嵌入向量 和 词（字）典
    Returns:
        - words (bcolz.carray): 词（字）典列表（bcolz carray  具有和 numpy array 类似的接口）
        - embeddings (bcolz.carray): 嵌入矩阵，每 1 行为 1 个 词向量/字向量，
                                     其行号即为该 词（字） 在 words 中的索引编号
    """
    folder_path = folder_path.rstrip('/')
    words = bcolz.carray(rootdir='%s/words'%folder_path, mode='r')
    embeddings = bcolz.carray(rootdir='%s/embeddings'%folder_path, mode='r')
    return words, embeddings

def get_embed(embed_root):

    embed = {}
    words, embeddings = load_embeddings(embed_root)  # embeddings中的数据类型是numpy.float32
    for i, w in enumerate(words):
        embed[w] = embeddings[i]
    print("load word embedding finished")
    return embed

def get_news_map(item_map, content_word_f, senK, normal, embed_root):
    embed = get_embed(embed_root)

    newsMap = {} # news编号(int) -> matric(20 * 60)

    article_in = open(content_word_f, 'r', encoding='UTF-8')
    line = article_in.readline()
    while line:
        id = line.strip()

        # 注意： >20的只要前20个句子，<20的补成20个句子
        if id in item_map:
            newsNumber = item_map[id]

            news_mat = np.zeros((senK, 64), dtype=np.float32) # 每个news的matric   20 * 64
            line = article_in.readline().strip()
            if line == "": # 没有单词的新闻
                newsMap[newsNumber] = news_mat
            else:
                line = line[:-2]
                sentences = line.split("||")
                for (i,sen) in enumerate(sentences[:senK]):
                    sen_embed = np.zeros(64, dtype=np.float32) # 每个sentence的embedding 64    float32
                    words = sen[:-1].split(",")
                    for word in words:
                        sen_embed += embed[word] # embed: float32
                    sen_embed /= len(words)
                    news_mat[i] += sen_embed

                length = len(sentences)
                if length < senK: # 不足20，需要补全的
                    temp = np.sum(news_mat[:length],axis=0) / length
                    for k in range(length,senK):
                        news_mat[k] += temp

                if normal=="meanstd":
                    for sen in news_mat:
                        sen -= np.mean(sen, axis=0)
                        sen /= np.std(sen,axis = 0)
                elif normal=="minmax":
                    for (i, sen) in enumerate(news_mat):
                        x_max = np.max(sen)
                        x_min = np.min(sen)
                        news_mat[i] = (sen - x_min) / (x_max - x_min)
                else:
                    pass

                newsMap[newsNumber] = news_mat

            line = article_in.readline()
        else:
            line = article_in.readline()
            line = article_in.readline()

    article_in.close()
    print("newsMap size:"+str(len(newsMap)))
    return newsMap


def get_news_map_doc2vec(item_map, senK, normal, content_word_f, content_word_w, doc2vec_model, hidden_dim):
    model_dm = Doc2Vec.load(doc2vec_model)
    if os.path.exists(content_word_w):
        fr = open(content_word_w, 'r')
        tmp = json.load(fr)
        newsMap = {}
        for key, value in tmp.items():
            newsMap[int(key)] = np.asarray(value)
        print("newsMap size 1:" + str(len(newsMap)))
        return  newsMap
    else:
        newsMap = {} # news编号(int) -> matric(20 * 60)
        newsMap_2 = {} # news编号(int) -> matric(20 * 60)

        article_in = open(content_word_f, 'r', encoding='UTF-8')
        line = article_in.readline()
        while line:
            id = line.strip()

            if id in item_map:
                newsNumber = item_map[id]

                news_mat = np.zeros((senK, hidden_dim), dtype=np.float32) # 每个news的matric   20 * 64
                line = article_in.readline().strip()
                if line == "": # 没有单词的新闻
                    newsMap[newsNumber] = news_mat
                    newsMap_2[newsNumber] = news_mat.tolist()
                else:
                    line = line[:-2]
                    sentences = line.split("||")
                    for (i,sen) in enumerate(sentences[:senK]):
                        sen_embed = model_dm.docvecs[str(id)+'-'+str(i)]
                        news_mat[i] += sen_embed

                    length = len(sentences)
                    if length < senK: # 不足20，需要补全的，使用真实句子的均值,在右边
                        temp = np.sum(news_mat[:length],axis=0) / length
                        for k in range(length,senK):
                            news_mat[k] += temp

                    if normal=="meanstd":
                        for sen in news_mat:
                            sen -= np.mean(sen, axis=0)
                            sen /= np.std(sen,axis = 0)
                    elif normal=="minmax":
                        for (i, sen) in enumerate(news_mat):
                            x_max = np.max(sen)
                            x_min = np.min(sen)
                            news_mat[i] = (sen - x_min) / (x_max - x_min)
                    else:
                        pass

                    newsMap[newsNumber] = news_mat
                    newsMap_2[newsNumber] = news_mat.tolist()

                line = article_in.readline()
            else:
                line = article_in.readline()
                line = article_in.readline()

        article_in.close()
        print("newsMap size 2:"+str(len(newsMap)))
        fw = open(content_word_w, 'w', encoding='utf-8')
        json.dump(newsMap_2, fw)
        return newsMap


def get_elements(item_map, element_f, element_root_w):
    if os.path.exists(element_root_w):
        fr = open(element_root_w, 'r')
        tmp = json.load(fr)
        newsMap = {}
        for key, value in tmp.items():
            newsMap[int(key)] = np.asarray(value)
        print("newsMap(element) size 1:" + str(len(newsMap)))
        return  newsMap
    else:
        newsMap = {} # news编号（int）--> matric(4 * 64) [time, person|organ, location, keywords]
        newsMap_2 = {} # news编号（int）--> matric(4 * 64) [time, person|organ, location, keywords]
        element_embed = open(element_f, 'r', encoding='utf-8').readlines()
        line_num = len(element_embed)
        i = 0
        while 8 * i < line_num:
            one_line = element_embed[8 * i: 8 * (i + 1)]
            element_dic = {}
            for ele in one_line:
                ele = ele.strip().split(':')
                element_dic[ele[0]] = ele[1]
            id = element_dic['id']
            time = element_dic['time']
            per = element_dic['per']
            organ = element_dic['organ']
            loc = element_dic['loc']
            keywords = element_dic['keywords']
            all = element_dic['all']
            if  id in item_map:
                f = False
                newsNumber = item_map[id]
                no_entity_set = set()
                news_mat = np.zeros((5, 64), dtype=np.float32)

                if time != '':
                    f = True
                    time = list(map(float, time.split(' ')))
                    news_mat[0] += time
                else:
                    no_entity_set.add(0)

                if per != '':
                    f = True
                    person = list(map(float, per.split(' ')))
                    news_mat[1] += person
                else:
                    no_entity_set.add(1)

                if organ != '':
                    f = True
                    organ = list(map(float, organ.split(' ')))
                    news_mat[2] += organ
                else:
                    no_entity_set.add(2)

                if loc != '':
                    f = True
                    location = list(map(float, loc.split(' ')))
                    news_mat[3] += location
                else:
                    no_entity_set.add(3)

                if keywords != '':
                    f = True
                    keywords = list(map(float, keywords.split(' ')))
                    news_mat[4] += keywords
                else:
                    no_entity_set.add(4)

                if all != '':
                    f = True
                    all_entity = list(map(float, all.split(' ')))
                    for no in no_entity_set:
                        news_mat[no] += all_entity

                if f:# 如果不是全为0，就进行正则化
                    for sen in news_mat:
                        sen -= np.mean(sen, axis=0)
                        sen /= np.std(sen, axis=0)
                else:# 如果全为0，就使之全为1
                    news_mat = np.ones((5, 64), dtype=np.float32)
                    print('have no entity'+str(id))
                newsMap[newsNumber] = news_mat
                newsMap_2[newsNumber] = news_mat.tolist()
            else:
                continue
            i += 1
        print("newsMap(element) size 2:" + str(len(newsMap)))
        fw = open(element_root_w, 'w', encoding='utf-8')
        json.dump(newsMap_2, fw)
        return newsMap
