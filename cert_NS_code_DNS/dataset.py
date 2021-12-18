# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import Dataset
from time import time
import random

N = 50

class ReadingNEWSTrain(Dataset):
    def __init__(self, sequences, targets, targets_time, newsMap = None,
                 elementsMap = None, usercandidate = None, negs=3, senK=20):
        self.data_hist = sequences.sequences
        self.data_hist_time = sequences.sequences_time
        self.data_cand = targets
        self.data_cand_time = targets_time
        self.data_user = sequences.user_ids
        self.newsMap = newsMap
        self.elementsMap = elementsMap
        self.usercandidate = usercandidate
        self.negs=negs
        self.senK=senK

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

        ####### 负样本新闻内容向量 & 负样本新闻元素向量 #########
        news_neg_can = np.zeros((N, 64), dtype=np.float32)  # 2D : 3 * 64
        # 原始的model是N，5，64，加DNS要改成N，64
        # news_element_neg_can = np.zeros((N, 5, 64), dtype=np.float32)  # 2D : 3 * 64
        news_element_neg_can = np.zeros((N, 64), dtype=np.float32)  # 2D : 3 * 64
        news_neg_id_can = np.zeros(N, dtype=np.int64)

        x = self.usercandidate[user]
        # 这一句不能注释，因为希望每个step的label都不一样
        # items = np.random.choice(a=x, size=N, replace=True, p=None)
        # shuffle(x)
        # 1
        # items = random.sample(x, N)
        # 2.
        m = random.randint(0, len(x) - N)
        items = x[m: m+N]
        for i, item in enumerate(items):
            news_neg_id_can[i] += item

            news = self.newsMap[item]
            news_temp = np.mean(news, axis=0, dtype=np.float32)
            news_neg_can[i] += news_temp

            news_element_temp = np.mean(self.elementsMap[item], axis=0)
            # news_element_temp = self.elementsMap[item] # 5,d
            news_element_neg_can[i] += news_element_temp
        news_neg_can = torch.from_numpy(news_neg_can)
        news_element_neg_can = torch.from_numpy(news_element_neg_can)
        news_neg_id_can = torch.from_numpy(news_neg_id_can)

        return news_hist, news_element_hist, history, \
               news_cand, news_element_cand, candidate, \
               user, hitory_time, cadidate_time, \
               news_neg_can, news_element_neg_can, news_neg_id_can

    def __len__(self):
        return len(self.data_hist)


class ReadingNEWSTest(Dataset):
    def __init__(self, sequences, targets, targets_time, newsMap=None,
                 elementsMap=None, usercandidate=None, negs=99, senK=20):
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
        # news_cand = np.zeros(64, dtype=np.float32)  # 1D : 64
        # for sen in news:
        #     news_cand += sen
        # news_cand = news_cand/len(news)
        news_cand = np.mean(news, axis=0, dtype=np.float32)
        news_cand = torch.from_numpy(news_cand)

        news_element_cand = self.elementsMap[candidate]
        news_element_cand = torch.from_numpy(news_element_cand)

        ####### 负样本新闻内容向量 & 负样本新闻元素向量 #########
        news_neg = np.zeros((self.negs, 64), dtype=np.float32) # 2D : 3 * 64
        news_element_neg = np.zeros((self.negs, 5, 64), dtype=np.float32) # 2D : 3 * 64
        news_neg_id = np.zeros(self.negs, dtype=np.int64)
        x = self.usercandidate[user]
        for i in range(self.negs):
            item = x[np.random.randint(len(x))]
            news_neg_id[i] += item

            news = self.newsMap[item]
            # sen_sum = np.zeros(64, dtype=np.float32)
            # for sen in news:
            #     sen_sum += sen
            # news_temp = sen_sum/len(news)
            news_temp = np.mean(news, axis=0, dtype=np.float32)
            news_neg[i] += news_temp
            # 这个地方不能均值
            news_element_temp = self.elementsMap[item] # 5，d
            news_element_neg[i] += news_element_temp

        news_neg = torch.from_numpy(news_neg)
        news_element_neg = torch.from_numpy(news_element_neg)
        news_neg_id = torch.from_numpy(news_neg_id)

        return news_hist, news_element_hist, history, news_cand, news_element_cand, candidate, \
               news_neg, news_element_neg, news_neg_id, user, hitory_time, cadidate_time

    def __len__(self):
        return len(self.data_hist)