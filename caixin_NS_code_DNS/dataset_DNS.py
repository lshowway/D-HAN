# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset

N = 50

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


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

        items = self.usercandidate[user]
        # 这一句不能注释，因为希望每个step的label都不一样
        items = np.random.choice(a=items, size=N, replace=True, p=None)

        for i, item in enumerate(items):
            news_neg_id_can[i] += item
            news = self.newsMap[item]
            news_temp = np.mean(news, axis=0)
            news_neg_can[i] += news_temp

            news_element_temp = np.mean(self.elementsMap[item], axis=0) # 64,这个地方要均值
            news_element_neg_can[i] += news_element_temp

        news_neg_can = torch.from_numpy(news_neg_can)
        news_element_neg_can = torch.from_numpy(news_element_neg_can)
        news_neg_id_can = torch.from_numpy(news_neg_id_can)

        return news_hist, news_element_hist, history, news_cand, news_element_cand, candidate, \
               user, hitory_time, cadidate_time, \
               news_neg_id_can, news_neg_can, news_element_neg_can

    def __len__(self):
        return len(self.data_hist)


class ReadingNEWSTest(Dataset):
    def __init__(self, args, sequences, targets, targets_time, newsMap=None, newsid_prob=None, news_latest_time=None,
                 elementsMap=None, usercandidate=None, negs = 3, senK = 20, sampling='static', state='train'):
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
        self.sampling = sampling
        self.news_popularity = newsid_prob
        self.news_lasted_time = news_latest_time
        self.state = state

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

        ####### 负样本新闻内容向量 & 负样本新闻元素向量 #########
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

            news_element_temp = self.elementsMap[item] # 4，64不可以均值
            news_element_neg[i] += news_element_temp

        news_neg = torch.from_numpy(news_neg)
        news_element_neg = torch.from_numpy(news_element_neg)
        news_neg_id = torch.from_numpy(news_neg_id)

        return news_hist, news_element_hist, history, news_cand, news_element_cand, candidate, \
               news_neg, news_element_neg, news_neg_id, user, hitory_time, cadidate_time

    def __len__(self):
        return len(self.data_hist)