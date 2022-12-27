# -*- coding: utf-8 -*-
import time
import torch
from torch import nn
import numpy as np
import argparse
from random import shuffle
from data_utils import generate_candidate
from process_text import *
from interactions import Interactions
from dataset_DNS import ReadingNEWS, ReadingNEWSTest
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW

from dna_ele_sen_news_tran import Tacnn


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class Recommender():
    def __init__(self, n_iter=None, neg_samples=None, neg_samples_test=None, learning_rate=None,
                 l2=None, optimizers=None, t_total=None, model_args=None):
        # model related
        self._num_users = None
        self._net = None
        self.model_args = model_args

        # learning related
        self._n_iter = n_iter
        self._neg_samples = neg_samples
        self._neg_samples_test = neg_samples_test
        self._learning_rate = learning_rate
        self.weight_decay = l2
        self.warmup_steps = model_args.warmup_steps
        self.t_total = t_total
        self._optimizers = optimizers

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):
        self._num_users = interactions.num_users
        self._num_items = interactions.num_items

        self._net = Tacnn(self.model_args, self._num_users, self._num_items).to(device)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self._net.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {"params": [p for n, p in self._net.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self._optimizer = AdamW(optimizer_grouped_parameters, lr=self._learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=self.warmup_steps * self.t_total, num_training_steps=self.t_total
        )

        para = sum([np.prod(list(p.size())) for p in self._net.parameters()])
        print("模型参数个数:" + str(para), flush=True)
        print('模型参数大小: {:4f}M'.format(para * 8 / 1000 / 1000), flush=True)

    def fit(self, train_loader, test_loader):
        for epoch_num in range(0, self._n_iter): # epoch
            t1 = time.time()
            epoch_loss = self.train_epoch(train_loader)
            t2 = time.time()
            if (epoch_num) % 1 == 0:
                HR, NDCG = self.test_epoch(test_loader, ks=10, epoch_num=epoch_num)
                output_str = "Epoch %d [%.1f s]\tloss=%.4f," \
                             "HR@1=%.4f,HR@2=%.4f,HR@3=%.4f,HR@4=%.4f,HR@5=%.4f,HR@6=%.4f,HR@7=%.4f,HR@8=%.4f,HR@9=%.4f,HR@10=%.4f, " \
                             "NDCG@1=%.4f,NDCG@2=%.4f,NDCG@3=%.4f,NDCG@4=%.4f,NDCG@5=%.4f,NDCG@6=%.4f,NDCG@7=%.4f,NDCG@8=%.4f,NDCG@9=%.4f,NDCG@10=%.4f, [%.1f s]" % (
                             epoch_num + 1,
                             t2 - t1,
                             epoch_loss,
                             HR[0], HR[1], HR[2], HR[3], HR[4], HR[5], HR[6], HR[7], HR[8], HR[9],
                             NDCG[0], NDCG[1], NDCG[2], NDCG[3], NDCG[4], NDCG[5], NDCG[6], NDCG[7], NDCG[8], NDCG[9],
                             time.time() - t2)
                print(output_str, flush=True)

    def train_epoch(self,train_loader):
        self._net.train()
        epoch_loss = 0
        for batch_idx, (news_hist, news_hist_ele, news_hist_id,
                        target_news, target_ele, target_news_id,
                        user, hitory_time, target_time,
                        can_id_list, can_embedding, can_ele_embedding,
                        ) in enumerate(train_loader):
            # train pos model
            target_prediction, items_all, loss_ns = self._net(news_hist.to(device),
                                      news_hist_ele.to(device),
                                      news_hist_id.to(device),
                                      target_news.to(device),
                                      target_ele.float().to(device),
                                      target_news_id.to(device),
                                      user.to(device),
                                      hitory_time.to(device),
                                      target_time.to(device),
                                                      True,
                                            can_id_list.to(device),
                                            can_embedding.to(device),
                                          can_ele_embedding.to(device),
                                      )
            # DNS
            items_all = items_all.tolist()

            neg_id_list = np.zeros((len(user), self._neg_samples), dtype=np.int64)  # batch, N
            neg_embedding = np.zeros((len(user), self._neg_samples, 64), dtype=np.float32)
            neg_ele_embedding = np.zeros((len(user), self._neg_samples, 5, 64), dtype=np.float32)

            for k, u in enumerate(user.tolist()):
                items = items_all[k]
                for i, item in enumerate(items):
                    neg_id_list[k, i] += item
                    news = newsMap[item]
                    news_temp = np.mean(news, axis=0)
                    neg_embedding[k, i] += news_temp

                    news_element_temp = elementsMap[item]
                    neg_ele_embedding[k, i] += news_element_temp

            neg_id_list = torch.from_numpy(neg_id_list)
            neg_embedding = torch.from_numpy(neg_embedding)
            neg_ele_embedding = torch.from_numpy(neg_ele_embedding)

            # DNS
            results = []
            for i in range(self._neg_samples):
                negative_prediction = self._net(news_hist.to(device),
                                                news_hist_ele.to(device),
                                                news_hist_id.to(device),
                                                neg_embedding[:, i, :].to(device),
                                                neg_ele_embedding[:, i, :].to(device),
                                                neg_id_list[:, i].to(device),
                                                user.to(device),
                                                hitory_time.to(device),
                                                target_time.to(device),
                                                train=False)

                results.append(negative_prediction)
            neg_pre = torch.cat(results, 1)

            # 损失函数
            target_temp = torch.clamp(torch.sigmoid(target_prediction), min=1e-10, max=1-1e-10)
            neg_temp = torch.clamp(1 - torch.sigmoid(neg_pre), min=1e-10, max=1-1e-10)
            positive_loss = -torch.mean(torch.log(target_temp))
            negative_loss = -torch.mean(torch.log(neg_temp))
            loss = 1.0 * positive_loss + 1.0 * negative_loss + 1.0 * loss_ns
            epoch_loss += loss.item()

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)

            # 更新参数
            self._optimizer.step()
            # self.scheduler.step()
            # 反向传播
            self._optimizer.zero_grad()

            # if batch_idx > 0:
            #     break

        epoch_loss /= (batch_idx + 1)

        return epoch_loss

    def test_epoch(self, test_loader, ks, epoch_num):
        self._net.eval()
        hr = [0] * ks
        ndcg = [0] * ks
        count = 0
        for batch_idx, (news_hist, news_hist_element, news_hist_id,
                        news_cand, news_cand_element, news_cand_id,
                        news_neg, news_neg_element, news_neg_id,
                        user, hitory_time, candidate_time) in enumerate(test_loader):  # batch
            target_prediction = self._net(news_hist.to(device),
                                          news_hist_element.to(device),
                                          news_hist_id.to(device),
                                          news_cand.to(device),
                                          news_cand_element.float().to(device),
                                          news_cand_id.to(device),
                                          user.to(device),
                                          hitory_time.to(device),
                                          candidate_time.to(device),
                                          train=False)# 已验证：torch.Size([batch, 1])
            results = []
            config.fw_sen = None
            for i in range(self._neg_samples_test):
                negative_prediction = self._net(news_hist.to(device),
                                                news_hist_element.to(device),
                                                news_hist_id.to(device),
                                                news_neg[:, i, :].to(device),
                                                news_neg_element[:, i, :].to(device),
                                                news_neg_id[:, i].to(device),
                                                user.to(device),
                                                hitory_time.to(device),
                                                candidate_time.to(device),
                                                train=False)
                results.append(negative_prediction.cpu().detach().numpy())
            results = [torch.from_numpy(t) for t in results]
            neg_pre = torch.cat(results, 1).to(device)  # 已验证：torch.Size([batch, 99])

            predictions = torch.cat((neg_pre, target_prediction), 1) # batch,100
            predictions = np.argsort(-predictions.cpu().detach().numpy(), axis=1)
            for i in range(len(predictions)): # 循环batch次
                count += 1
                oneline = predictions[i,:]
                for k in range(ks):
                    rec = oneline[:k + 1]
                    if 99 in rec:
                        hr[k] += 1
                    for pos in range(k + 1):
                        if rec[pos] == 99:
                            ndcg[k] += 1 / np.log2(1 + pos + 1)
        HR = []
        NDCG = []
        for k in range(ks):
            HR.append(float(hr[k]) / float(count))
            NDCG.append(float(ndcg[k]) / float(count))
        config.fw_sen = None
        return HR, NDCG


def get_args():
    parser = argparse.ArgumentParser()
    # caixin
    parser.add_argument('--train_root', type=str, default='../caixin/userSeq_train(L=10).txt')
    parser.add_argument('--test_root', type=str, default='../caixin/userSeq_test(L=10).txt')
    parser.add_argument('--content_word', type=str, default='../caixin/newsid_word(L=10).txt')
    parser.add_argument('--content_word_w', type=str, default='../caixin/newsid_word(L=10).vec')
    parser.add_argument('--embed_root', type=str, default='../../zh.64/')
    parser.add_argument('--element_root', type=str, default='../caixin/newsid_embed_caixin.txt')
    parser.add_argument('--element_root_w', type=str, default='../caixin/newsid_embed_caixin.vec')
    parser.add_argument('--doc2vec_model', type=str, default='../caixin/doc2vec_caixin')
    ### caixin

    parser.add_argument('--L', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)  # 256
    parser.add_argument('--batch_size_test', type=int, default=4)  # 10
    parser.add_argument('--optimizers', type=str, default="adam")  # adam\sgd\adadelta
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument("--warmup_steps", default=0.0, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--n_iter', type=int, default=100)  # 100
    parser.add_argument('--normalization', type=str, default="meanstd")  # meanstd\minmax\none
    parser.add_argument('--senK', type=int, default=20)

    parser.add_argument('--pop_neg', type=int, default=10)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--neg_samples_test', type=int, default=99)
    parser.add_argument('--interval_or_abs', type=str, default='none')
    parser.add_argument('--sampling', type=str, default='static')
    # parser.add_argument('--sen_case_study', type=str, default='../caixin/time_news_weights.txt')
    parser.add_argument('--theta', type=float, default=0.3)
    # v6
    parser.add_argument('--kernel_sizes', type=int, default=3)
    parser.add_argument('--kernel_num', type=int, default=64)

    parser.add_argument('--user_dim', type=int, default=64)
    parser.add_argument('--news_dim', type=int, default=64)
    parser.add_argument('--element_dim', type=int, default=64)
    # v2,V6,V7
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_attention_heads', type=int, default=4)
    parser.add_argument('--intermediate_size', type=int, default=256)

    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--time_factor', type=float, default=0.01)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.2)
    parser.add_argument('--hidden_act', type=str, default='gelu')
    config = parser.parse_args()

    return config


if __name__ == '__main__':
    # cpu:time为int32; gpu:time为int64(interactions.py)
    # gpu中加大test_batch
    config = get_args()
    config.L = config.L
    # config.fw_sen = None
    print("model_config:"+str(config), flush=True)
    USE_CUDA = torch.cuda.is_available() and True
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print('using .... ', device)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    # load train(test)|test dataset
    train = Interactions(config.train_root)
    train.to_sequence(config.L)
    test = Interactions(config.test_root, user_map=train.user_map, item_map=train.item_map)
    print("train.num_users:" + str(train.num_users) + ",train.num_items:" + str(train.num_items), flush=True)
    print("test.num_users:" + str(test.num_users) + ",test.num_items:" + str(test.num_items), flush=True)

    # load news content |cal candidate neg samples
    newsMap = get_news_map_doc2vec(test.item_map, config.senK, config.normalization,
                                   config.content_word, config.content_word_w, config.doc2vec_model, hidden_dim=config.news_dim) # 20*64
    elementsMap = get_elements(test.item_map, config.element_root, config.element_root_w)
    train_candidate, test_candidate = generate_candidate(train)  # 这是什么？

    train_dataset = ReadingNEWS(args=config, sequences=train.sequences, targets=train.sequences.targets, targets_time=train.sequences.targets_time,
                                newsMap=newsMap, elementsMap=elementsMap,
                                usercandidate=train_candidate, negs=config.neg_samples, senK=config.senK)
    test_dataset = ReadingNEWSTest(args=config, sequences=train.test_sequences, targets=test.item_ids, targets_time=test.timestamps,
                               newsMap=newsMap, elementsMap=elementsMap,
                               usercandidate=test_candidate, negs=config.neg_samples_test, senK=config.senK)
    print("Define dataset finished.", flush=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size_test,shuffle=False, pin_memory=False)
    print("Define dataloader finished, # train batch: %s %s, # test batch: %s %s"
          %(len(train_loader), len(train_loader)*config.batch_size, len(test_loader), len(test_loader)*config.batch_size_test), flush=True)

    # Set up the network and training parameters
    t_total = len(train_loader) * config.n_iter
    model = Recommender(n_iter=config.n_iter,
                        neg_samples=config.neg_samples,
                        neg_samples_test = config.neg_samples_test,
                        learning_rate=config.learning_rate,
                        l2=config.l2,
                        optimizers=config.optimizers,
                        t_total=t_total,
                        model_args=config)
    model._initialize(test)
    print("Network initial finished.", flush=True)
    model.fit(train_loader, test_loader)
