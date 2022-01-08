import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import json
import torch
from gensim.models.doc2vec import Doc2Vec




def generate_candidate(file_path):
    train_candidate = dict()
    test_candidate = dict()
    fin = open(file_path, 'r', encoding='UTF-8')
    line = fin.readline().strip()
    while line:
        id = line
        line = fin.readline().strip()
        all_candidates = list(map(int, line.split(' ')))
        train_candidate[int(id)] = np.random.choice(a=all_candidates, size=2000, replace=True, p=None)
        test_candidate[int(id)] = all_candidates
        line = fin.readline().strip()
    return train_candidate, test_candidate





def get_news_map_doc2vec(item_map, senK, normal, content_word_f, content_word_w, doc2vec_model, hidden_dim):
    model_dm = Doc2Vec.load(doc2vec_model)
    if os.path.exists(content_word_w):
        fr = open(content_word_w, 'r')
        tmp = json.load(fr)
        newsMap = {}
        for key, value in tmp.items():
            newsMap[int(key)] = np.asarray(value)
        print("newsMap size:" + str(len(newsMap)))
        return  newsMap
    else:
        newsMap = {} # news id (int) -> matrix(20 * 60)
        newsMap_2 = {} # news id (int) -> matrix(20 * 60)

        article_in = open(content_word_f, 'r', encoding='UTF-8')
        line = article_in.readline()
        while line:
            id = line.strip()

            if id in item_map:
                newsNumber = item_map[id]

                news_mat = np.zeros((senK, hidden_dim), dtype=np.float32) # 每个news的matric   20 * 64
                line = article_in.readline().strip()
                if line == "": # news of no words
                    newsMap[newsNumber] = news_mat
                    newsMap_2[newsNumber] = news_mat.tolist()
                else:
                    line = line[:-2]
                    sentences = line.split("||")
                    for (i,sen) in enumerate(sentences[:senK]):
                        sen_embed = model_dm.docvecs[str(id)+'-'+str(i)]
                        news_mat[i] += sen_embed

                    length = len(sentences)
                    if length < senK: # pad to 20 on the right, mean value is used
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
        print("newsMap size:"+str(len(newsMap)))
        fw = open(content_word_w, 'w', encoding='utf-8')
        json.dump(newsMap_2, fw)
        return newsMap


def get_elements(item_map, element_f, element_root_w):
    if os.path.exists(element_root_w):
        fr = open(element_root_w, 'r')
        tmp = json.load(fr)
        newsMap = {}
        for key, value in tmp.items():
            newsMap[int(key)] = np.asarray(value, dtype=np.float32)
        print("newsMap(element) size:" + str(len(newsMap)))
        return  newsMap
    else:
        newsMap = {} # news id（int）--> matric(4 * 64) [time, person|organ, location, keywords]
        newsMap_2 = {} # news id（int）--> matric(4 * 64) [time, person|organ, location, keywords]
        element_embed = open(element_f, 'r', encoding='utf-8').readlines()
        line_num = len(element_embed)
        i = 0
        while 8 * i < line_num:
            if i % 10000 == 0:
                print(i)
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
            if id in item_map:
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

                if f: # If not all are 0, regularization is performed
                    for sen in news_mat:
                        sen -= np.mean(sen, axis=0)
                        sen /= np.std(sen, axis=0)
                else: # If all are 0, set to 1
                    news_mat = np.ones((5, 64), dtype=np.float32)
                    print('have no entity'+str(id))
                newsMap[newsNumber] = news_mat
                newsMap_2[newsNumber] = news_mat.tolist()
            else:
                continue
            i += 1
        print("newsMap(element) size:" + str(len(newsMap)))
        fw = open(element_root_w, 'w', encoding='utf-8')
        json.dump(newsMap_2, fw)
        return newsMap




N = 50  # this is used for DNS. 2. DNS: random sample 50 samples, and then use DNS to sample neg_sample
class ReadingNEWS(Dataset):
    def __init__(self, args, sequences, targets, targets_time, newsMap=None,
                 elementsMap=None, usercandidate=None, negs=3, senK=20):
        self.args = args
        self.data_hist = sequences.sequences
        self.data_hist_time = sequences.sequences_time
        self.data_cand = targets
        self.data_cand_time = targets_time
        self.data_user = sequences.user_ids
        self.newsMap = newsMap
        self.elementsMap = elementsMap
        self.usercandidate = usercandidate
        self.negs = negs
        self.senK = senK

    def __getitem__(self, index):  # 这个index
        history = self.data_hist[index]             # 长度L
        hitory_time = self.data_hist_time[index]    # 长度L
        candidate = self.data_cand[index]           # 长度1
        cadidate_time = self.data_cand_time[index]  # 长度1
        user = self.data_user[index]                # 长度1

        ####### 历史新闻内容向量 & 历史新闻元素向量 #######
        news_hist = np.zeros((history.shape[0], self.senK, 64), dtype=np.float32)  # 3 D ：L * 20 * 64
        news_element_hist = np.zeros((history.shape[0], 5, 64), dtype=np.float32)  # 3 D ：L * 4 * 64
        for (i, item) in enumerate(history):
            news = self.newsMap[item]
            news_hist[i,:,:] += news
            news_element = self.elementsMap[item]
            news_element_hist[i,:,:] += news_element
        news_hist = torch.from_numpy(news_hist)
        news_element_hist = torch.from_numpy(news_element_hist)
        ######## 候选新闻内容向量 & 候选新闻元素向量 ########
        news = self.newsMap[candidate]
        news_cand = np.mean(news, axis=0, dtype=np.float32)
        news_cand = torch.from_numpy(news_cand)

        news_element_cand = self.elementsMap[candidate]
        news_element_cand = torch.from_numpy(news_element_cand)

        ####### 候选负样本新闻内容向量 & 负样本新闻元素向量 #########
        news_neg_can = np.zeros((N, 64), dtype=np.float32) # 2D : 3 * 64
        news_element_neg_can = np.zeros((N, 64), dtype=np.float32) # 2D : 3 * 64
        news_neg_id_can = np.zeros(N, dtype=np.int64)

        x = self.usercandidate[user]
        items = np.random.choice(a=x, size=N, replace=True, p=None)

        for i, item in enumerate(items):
            news_neg_id_can[i] += item
            news = self.newsMap[item]
            news_temp = np.mean(news, axis=0)
            news_neg_can[i] += news_temp

            news_element_temp = np.mean(self.elementsMap[item], axis=0)
            news_element_neg_can[i] += news_element_temp

        news_neg_can = torch.from_numpy(news_neg_can)
        news_element_neg_can = torch.from_numpy(news_element_neg_can)
        news_neg_id_can = torch.from_numpy(news_neg_id_can)

        return (news_hist, news_element_hist, history, news_cand, news_element_cand, candidate, \
               user, hitory_time, cadidate_time, \
               news_neg_id_can, news_neg_can, news_element_neg_can)

    def __len__(self):
        return len(self.data_hist)


class ReadingNEWSTest(Dataset):
    def __init__(self, args, sequences, targets, targets_time, newsMap=None,
                 elementsMap=None, usercandidate=None, negs=3, senK=20):
        self.args = args
        self.data_hist = sequences.sequences
        self.data_hist_time = sequences.sequences_time
        self.data_cand = targets
        self.data_cand_time = targets_time
        self.data_user = sequences.user_ids
        self.newsMap = newsMap
        self.elementsMap = elementsMap
        self.usercandidate = usercandidate
        self.negs = negs
        self.senK = senK

    def __getitem__(self, index):  # 这个index
        history = self.data_hist[index]             # 长度L
        hitory_time = self.data_hist_time[index]    # 长度L
        candidate = self.data_cand[index]           # 长度1
        cadidate_time = self.data_cand_time[index]  # 长度1
        user = self.data_user[index]                # 长度1

        ####### 历史新闻内容向量 & 历史新闻元素向量 #######
        news_hist = np.zeros((history.shape[0], self.senK, 64), dtype=np.float32)  # 3 D ：L * 20 * 64
        news_element_hist = np.zeros((history.shape[0], 5, 64), dtype=np.float32)  # 3 D ：L * 4 * 64
        for (i, item) in enumerate(history):
            news = self.newsMap[item]
            news_hist[i,:,:] += news
            news_element = self.elementsMap[item]
            news_element_hist[i,:,:] += news_element
        news_hist = torch.from_numpy(news_hist)
        news_element_hist = torch.from_numpy(news_element_hist)
        ######## 候选新闻内容向量 & 候选新闻元素向量 ########
        news_cand = np.zeros(64, dtype=np.float32)         # 1D : 64
        news = self.newsMap[candidate]
        for sen in news:
            news_cand += sen
        news_cand = news_cand/len(news)
        news_cand = torch.from_numpy(news_cand)

        news_element_cand = self.elementsMap[candidate]
        news_element_cand = torch.from_numpy(news_element_cand)

        ####### negative news item representation, and its element rep
        news_neg = np.zeros((self.negs, 64), dtype=np.float32) # 2D : 3 * 64
        news_element_neg = np.zeros((self.negs, 5, 64), dtype=np.float32) # 2D : 3 * 64
        news_neg_id = np.zeros(self.negs, dtype=np.int64)

        x = self.usercandidate[user]
        items = np.random.choice(a=x, size=self.negs, replace=True, p=None)

        for i, item in enumerate(items):
            news_neg_id[i] += item

            news = self.newsMap[item]
            sen_sum = np.zeros(64, dtype=np.float32)
            for sen in news:
                sen_sum += sen
            news_temp = sen_sum/len(news)
            news_neg[i] += news_temp

            news_element_temp = self.elementsMap[item]
            news_element_neg[i] += news_element_temp

        news_neg = torch.from_numpy(news_neg)
        news_element_neg = torch.from_numpy(news_element_neg)
        news_neg_id = torch.from_numpy(news_neg_id)

        return (news_hist, news_element_hist, history, news_cand, news_element_cand, candidate, \
                user, hitory_time, cadidate_time,
               news_neg_id, news_neg, news_element_neg)

    def __len__(self):
        return len(self.data_hist)
