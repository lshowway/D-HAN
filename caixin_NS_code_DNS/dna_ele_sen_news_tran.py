# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

USE_CUDA = torch.cuda.is_available() and True
device = torch.device("cuda" if USE_CUDA else "cpu")
from SelfAttention import ScaledDotSelfAttention, TransformerLayer


class SenSelfAttNet(nn.Module):
    def __init__(self, model_args):
        super(SenSelfAttNet, self).__init__()
        self.linear = nn.Linear(64, model_args.hidden_size)
        self.self_attn = ScaledDotSelfAttention(model_args)
        self.fc = nn.Linear(model_args.senK+2, 1)
        self.fc_2 = nn.Sequential(nn.Linear(model_args.hidden_size, model_args.news_dim))

    def forward(self, sen_embed, can_embed, user_embed):
        # sen_embed:batch,senK,dim  can_embed:batch,1, dim   user_embed: batch,1, dim
        can_embed = can_embed.unsqueeze(1)
        user_embed = user_embed.unsqueeze(1)
        self_attn_input = torch.cat((sen_embed, can_embed, user_embed), 1)  # 这里拼接的第1维，可以拼接第2维
        self_attn_input = self.linear(self_attn_input)  # batch, senK, 128
        attn_output = self.self_attn(self_attn_input).permute(0, 2, 1)  # batch,  128, senK
        attn_output = self.fc(attn_output).permute(0, 2, 1)  # batch, 1, 64
        attn_output = self.fc_2(attn_output)  # batch, 1, 768 -> batch, 1, 64
        return attn_output


class EleSelfAttNet(nn.Module):
    def __init__(self, model_args):
        super(EleSelfAttNet, self).__init__()
        self.linear = nn.Linear(64 * 2, model_args.hidden_size)
        self.attention = ScaledDotSelfAttention(model_args)
        self.fc = nn.Linear(5, 1)
        self.fc_2 = nn.Sequential(nn.Linear(model_args.hidden_size, model_args.news_dim))

    def forward(self, hist_ele_embed, can_ele_embed):
        # hist: batch, 5, dim, can: batch,5,dim
        input = torch.cat([hist_ele_embed, can_ele_embed], dim=2)  # batch,5,dim*2
        input = self.linear(input) # batch,5,hidden_size
        output = self.attention(input)  # batch,5,hidden_size
        output = output.permute(0, 2, 1) # batch,hidden_size,5

        vec = self.fc(output).permute(0, 2, 1) # batch, hidden_size, 1
        vec = self.fc_2(vec)  # batch, 1, hidden_size -> batch, 1, dim

        return vec


class NewsSelfAttNet(nn.Module):
    def __init__(self, model_args):
        super(NewsSelfAttNet, self).__init__()
        self.args = model_args
        self.linear = nn.Linear(model_args.news_dim * 6, model_args.hidden_size)
        self.linear_2 = nn.Linear(model_args.news_dim * 3, model_args.hidden_size)
        self.linear_3 = nn.Linear(model_args.news_dim * 5, model_args.hidden_size)
        self.attn = TransformerLayer(model_args)

    def forward(self, v, q, u, hitory_time, cadidate_time, turn='hist'):
        # 1. v: batch, L, dim*3, q:batch,dim*3, u: batch,dim, abs_embed: batch, L, dim
        # 2. v: batch, L, dim, q: batch, L, dim, ..
        # candidate_tie:batch,, history_time:batch,L

        if turn =='hist':
            q = q.unsqueeze(1).repeat(1, v.size(1), 1)
            input = torch.cat((v, q, ), 2) # 拼接 batch, L, dim
            input = self.linear(input)
            logits = self.attn(input)  # batch, L, dim

            seconds = (cadidate_time.unsqueeze(1).repeat(1, hitory_time.size(1)) - hitory_time).type(
                torch.FloatTensor)  # batch, L, 1
            hours = seconds.to(device) / 3600  # 越往后，离得越近
            interval = torch.exp((-self.args.time_factor * (hours)).unsqueeze(2))  # 后面是1.0，前面是0.9 batch, L, 1

            logits = torch.mul(logits, interval).unsqueeze(1)  # batch, 1, L, dim
        elif turn == 'candidate': # q: batch T d2
            # input = v
            input = torch.cat([v, q, u], dim=-1) # batch T 64*3
            input = self.linear_2(input)
            logits = self.attn(input)
        elif turn == 'target':
            input = torch.cat([v, q, u], dim=-1)  # batch  64*5
            input = self.linear_3(input)
            logits = self.attn(input)
        else:
            pass
        return logits


class TransLayerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([TransformerLayer(config) for _ in range(config.layer_num)])

    def forward(self, hidden_states,):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)

        return hidden_states


class Tacnn(nn.Module):
    def __init__(self, model_args, num_users, num_items):
        super(Tacnn, self).__init__()
        self.args = model_args
        self.num_items = num_items

        self.L = self.args.L
        self.user_dim = self.args.user_dim
        self.news_dim = self.args.news_dim
        self.element_dim = self.args.element_dim
        self.kernel_num = self.args.kernel_num
        self.kernel_sizes = self.args.kernel_sizes  # 卷积核高度
        self.layer_num = self.args.layer_num  # cnn层数
        self.time_factor = self.args.time_factor

        self.user_embeddings = nn.Embedding(num_users, self.user_dim)
        self.user_embeddings.weight.data.normal_(mean=0.0, std=0.02)  # 1.0 / self.user_embeddings.embedding_dim

        self.item_embeddings = nn.Embedding(num_items, self.news_dim)
        self.item_embeddings.weight.data.normal_(mean=0.0, std=0.02)  # 1.0 / self.item_embeddings.embedding_dim

        self.senatt_net = SenSelfAttNet(model_args).to(device)
        self.eleatt_net = EleSelfAttNet(model_args).to(device)
        self.newsatt_net = NewsSelfAttNet(model_args).to(device)

        self.trans = TransLayerEncoder(model_args)

        self.fc = nn.Sequential(
            nn.Linear(model_args.hidden_size * model_args.L + model_args.news_dim * 4, self.news_dim * 5),
            nn.ReLU(inplace=True),
            nn.Linear(self.news_dim * 5, int(self.news_dim * 2.5)),
            nn.ReLU(inplace=True),
            nn.Linear(int(self.news_dim * 2.5), 1))

        from dataset_DNS import N
        # self.W_1 = nn.Linear(model_args.L, 1)
        # self.W_2 = nn.Linear(N, N)

        self.W_x = nn.Linear(N, N)

        # self.W_a = nn.Linear(model_args.L, 1)

    def forward(self, x, x_element, x_id, \
                can_embed, can_element, can_id, \
                user_var, hitory_time, cadidate_time,
                train,
                can_id_list=None, can_embedding=None, can_ele_embedding=None,):
        # 用户ID embedding, 新闻ID embedding, 候选新闻ID embedding
        user_emb = self.user_embeddings(user_var)
        item_embs = self.item_embeddings(x_id)  # batch, L, 64
        can_id_emb = self.item_embeddings(can_id).squeeze(1)  # batch, 64

        # 1. 历史news句子表示，融入user，候选news信息
        # hist_embed: batch, L, senK, dim
        news_matrix = [self.senatt_net(x[:, i, :, :], can_embed, user_emb) for i in range(x.size(1))]
        news_matrix = torch.stack(news_matrix, 2).squeeze(1)  # batch, L, 64

        # 2. 历史news元素表示,hist_element: batch, L, 5, dim, can_element: batch, 5, dim
        news_element_matrix = [self.eleatt_net(x_element[:, i, :, :], can_element) for i in range(x_element.size(1))]
        news_element_matrix = torch.stack(news_element_matrix, 2).squeeze(1)  # batch, L, 64

        # 3. 历史news表示，包括：文本表示，id表示，元素表示
        news_matrix = torch.cat((news_matrix, item_embs, news_element_matrix), 2)  # batch, L, 192

        # 一、候选news元素表示
        can_element = torch.mean(can_element, dim=1)
        # 二、候选news表示，包括：文本表示，id表示，元素表示
        can_embed = torch.cat((can_embed, can_id_emb, can_element), 1)  # batch, 192

        # 三、历史新闻表示和候选新闻表示attention，外加user表示，加入时间间隔信息
        # batch，1, L, 128
        news_matrix_1 = self.newsatt_net(news_matrix, can_embed, user_emb, hitory_time, cadidate_time)

        # 输入到Transformer
        news_matrix = self.trans(news_matrix_1)

        news_vector = news_matrix.contiguous().view(news_matrix.size(0), -1)
        fc_input = torch.cat((news_vector, can_embed, user_emb), 1)
        output = self.fc(fc_input)
        if train:
            # 衡量history representation和candidate representation的相似度从而选择负样本
            # 第一种
            # can_id_embedding = self.item_embeddings(can_id_list)  # batch, 1000, 64
            # id_ele_news_user_embedding = torch.cat([can_embedding, can_id_embedding, can_ele_embedding], dim=-1)  # batch, T, 64*3
            # can_embedding = self.newsatt_net(id_ele_news_user_embedding, can_id_embedding, user_emb, hitory_time, cadidate_time)  # batch L
            # t1 = torch.matmul(news_matrix.squeeze(), can_embedding.permute(0, 2, 1)).squeeze() # batch L T
            # t1 = self.W_a(t1.permute(0, 2, 1)).permute(0, 2, 1).squeeze() # batch  T
            # t1 = nn.Softmax(dim=1)(t1)

            # 第二种
            # X1 = news_matrix_1.squeeze() # news attention 的输出
            # X1 = news_matrix.squeeze() # trans的输出
            # can_id_embedding = self.item_embeddings(can_id_list)  # batch, 1000, 64
            # X1 = self.W_1(X1.permute(0, 2, 1)).permute(0, 2, 1) # batch 1, d
            # X2 = self.newsatt_net(can_embedding, can_ele_embedding, can_id_embedding, None, None, turn='candidate') # batch T d
            # t1 = torch.matmul(X1, X2.permute(0, 2, 1)).squeeze() # batch T
            # t1 = self.W_2(t1) # batch T
            # t1 = nn.Softmax(dim=-1)(t1)

            # 第三种：candidates不是与hist相近，而是要与target（label）更相近
            target_id_embedding = can_id_emb # batch, 64 can_id_emb
            can_id_embedding = self.item_embeddings(can_id_list)  # batch, 1000, 64
            X1 = self.newsatt_net(can_embed, can_element, target_id_embedding, None, None, turn='target').unsqueeze(1)  # batch 1 d target rep
            X2 = self.newsatt_net(can_embedding, can_ele_embedding, can_id_embedding, None, None, turn='candidate')  # batch T d
            t1 = torch.matmul(X1, X2.permute(0, 2, 1)).squeeze() # batch T
            t1 = self.W_x(t1)
            t1 = nn.Softmax(dim=-1)(t1)

            values, indices = t1.topk(self.args.neg_samples, dim=1, largest=True, sorted=True)
            indices = torch.gather(can_id_list, dim=1, index=indices)

            i1 = torch.argmax(t1, dim=-1) # batch, 1
            X2_t = X2[range(i1.size(0)), i1]
            loss = torch.mul(X1, X2_t) # batch
            loss = torch.sigmoid(loss)
            loss = torch.clamp(loss, min=1e-10, max=1-1e-10)
            loss = torch.log(loss) # 这里只有一个loss？
            loss = -torch.mean(loss)

            # values, indices = t1.topk(self.args.neg_samples, dim=1, largest=True, sorted=True)
            # indices = torch.gather(can_id_list, dim=1, index=indices)

            # 衡量history representation和candidate representation的相似度从而选择负样本
            return output, indices, loss
        else:
            return output


