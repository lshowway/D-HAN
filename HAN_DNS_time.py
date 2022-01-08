# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

USE_CUDA = torch.cuda.is_available() and True
device = torch.device("cuda" if USE_CUDA else "cpu")
from Attentions import ScaledDotSelfAttention, TransformerLayer, TimeEmbedding


class SenSelfAttNet(nn.Module):
    def __init__(self, model_args):
        super(SenSelfAttNet, self).__init__()
        self.linear = nn.Linear(64, model_args.hidden_size)
        self.self_attn = ScaledDotSelfAttention(model_args)
        self.fc = nn.Linear(model_args.senK+2, 1)
        self.fc_2 = nn.Sequential(nn.Linear(model_args.hidden_size, model_args.news_dim))

    def forward(self, sen_embed, can_embed, user_embed):
        # 计算attention score sen_embed:batch,senK,dim  can_embed:batch,dim   user_embed: batch,dim
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
        self.attn = TransformerLayer(model_args)
        self.time_embed = TimeEmbedding(model_args)

        self.linear_2 = nn.Linear(model_args.news_dim * 3, model_args.hidden_size)
        self.linear_3 = nn.Linear(model_args.news_dim * 5, model_args.hidden_size)

        if model_args.interval_or_abs == 'abs':
            self.w_a = nn.Linear(model_args.news_dim * 3 + model_args.hidden_size * 1, model_args.hidden_size)
            self.w_b = nn.Linear(model_args.L + 1, model_args.L)
        elif model_args.interval_or_abs == 'interval':
            self.w_3 = nn.Linear(model_args.news_dim * 6 + model_args.hidden_size * 1, model_args.hidden_size)
        else:
            self.w_1 = nn.Linear(model_args.news_dim * 6 + model_args.hidden_size * 3, model_args.hidden_size)

    def forward(self, hist_embed, can_embed, user_embed, history_time, candidate_time, turn='hist'):
        """his_embed: batch, L, dim*3,   can_embed:batch,dim*3,    user_embed: batch,dim"""
        # 相对时间差V3
        if history_time is not None and candidate_time is not None:
            absolute_embedding, interval_embedding = self.time_embed(history_time, candidate_time)  # batch, L, hidden_size
            if self.args.interval_or_abs == 'interval':
                # print('===> user time interval')
                hist_embed = torch.cat((hist_embed, interval_embedding), 2)  # batch, L,dim*3+hidden_size
                can_embed = can_embed.unsqueeze(1).repeat(1, self.args.L, 1)  # batch, L, dim*3
                input = torch.cat([hist_embed, can_embed, ], dim=2)
                input = self.w_3(input)  # batch, L, hidden_size
                logits = self.attn(input)  # batch, L, dim
            # 绝对时间V2
            elif self.args.interval_or_abs == 'abs':
                # print('===> user absolute time stamp')
                can_embed = torch.cat([can_embed, absolute_embedding[:, -1, :]], dim=1).unsqueeze(1)  # batch, 1, dim*3 + hidden_size
                hist_embed = torch.cat([hist_embed, absolute_embedding[:, :-1, :]], dim=2)  # batch, L, dim*3+hidden_size
                input = torch.cat([hist_embed, can_embed], dim=1)  # batch, L+1, dim*3+hidden_size
                input = self.w_a(input)
                logits = self.attn(input)  # batch, L+1, hidden_size
                logits = self.w_b(logits.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                # print('===> user time interval and absolute time stamp')
                hist_embed = torch.cat((hist_embed, absolute_embedding[:, :-1, :], interval_embedding), 2)  # batch, L,dim*3+hidden_size*2
                can_embed = torch.cat([can_embed, absolute_embedding[:, -1, :]], dim=1).unsqueeze(1)  # batch, 1, dim*3 + hidden_size
                can_embed = can_embed.repeat(1, self.args.L, 1)  # batch, L, dim*3 + hidden_size
                input = torch.cat([hist_embed, can_embed], dim=2) # batch, L, (dim*3+hidden_size*2)2
                input = self.w_1(input)  # batch, L, hidden_size
                logits = self.attn(input)  # batch, L, dim

            return logits.unsqueeze(1) # batch, 1, L, dim
        else:
            if turn == 'candidate':  # q: batch T d2
                input = torch.cat([hist_embed, can_embed, user_embed], dim=-1)  # batch T 64*3
                input = self.linear_2(input)
                logits = self.attn(input)
            elif turn == 'target':
                input = torch.cat([hist_embed, can_embed, user_embed], dim=-1)  # batch  64*5
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


class D_HAN(nn.Module):
    def __init__(self, model_args, num_users, num_items):
        super(D_HAN, self).__init__()
        self.args = model_args
        self.num_items = num_items

        self.L = self.args.L
        self.num_negs = self.args.neg_samples
        self.user_dim = self.args.user_dim
        self.news_dim = self.args.news_dim
        self.element_dim = self.args.element_dim
        # self.kernel_num = self.args.kernel_num
        # self.kernel_sizes = self.args.kernel_sizes  # 卷积核高度
        # self.layer_num = self.args.layer_num  # cnn层数
        # self.time_factor = self.args.time_factor

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

        from utils import N
        self.W_x = nn.Linear(N, N)

    def forward(self, x, x_element, x_id, \
                can_embed, can_element, can_id, \
                user_var, history_time, candidate_time,
                train,
                can_id_list=None, can_embedding=None, can_ele_embedding=None,):
        # user ID embedding, news ID embedding, candidate news ID embedding
        user_emb = self.user_embeddings(user_var)
        item_embs = self.item_embeddings(x_id)  # batch, L, 64
        can_id_emb = self.item_embeddings(can_id).squeeze(1)  # batch, 64

        # 1. history news sentence representation，inject user，candidate news information
        # hist_embed: batch, L, senK, dim
        news_matrix = [self.senatt_net(x[:, i, :, :], can_embed, user_emb) for i in range(x.size(1))]
        news_matrix = torch.stack(news_matrix, 2).squeeze(1)  # batch, L, 64

        # 2. history news element representation,hist_element: batch, L, 5, dim, can_element: batch, 5, dim
        news_element_matrix = [self.eleatt_net(x_element[:, i, :, :], can_element) for i in range(x_element.size(1))]
        news_element_matrix = torch.stack(news_element_matrix, 2).squeeze(1)  # batch, L, 64

        # 3. history news representation，including：text representation，id representation，element representation
        news_matrix = torch.cat((news_matrix, item_embs, news_element_matrix), 2)  # batch, L, 192

        # a. candidate news element representation
        can_element = torch.mean(can_element, dim=1)
        # b. candidate news representation, including: text representation, id representation and element representation
        can_embed = torch.cat((can_embed, can_id_emb, can_element), 1)  # batch, 192

        # c. history news representation and candidate news rep, attention, with user rep, and time interval info
        # batch，1, L, 128
        news_matrix_1 = self.newsatt_net(news_matrix, can_embed, user_emb, history_time, candidate_time)

        news_matrix = self.trans(news_matrix_1)

        news_vector = news_matrix.contiguous().view(news_matrix.size(0), -1)
        fc_input = torch.cat((news_vector, can_embed, user_emb), 1)
        output = self.fc(fc_input)
        if train:
            # three kinds of negative sampling are tested, and the 3rd is adopted
            # compute the similarity between history and candidate representation,
            # and then select negative samples according to this similarity score
            # 1st
            # can_id_embedding = self.item_embeddings(can_id_list)  # batch, 1000, 64
            # id_ele_news_user_embedding = torch.cat([can_embedding, can_id_embedding, can_ele_embedding], dim=-1)  # batch, T, 64*3
            # can_embedding = self.newsatt_net(id_ele_news_user_embedding, can_id_embedding, user_emb, hitory_time, cadidate_time)  # batch L
            # t1 = torch.matmul(news_matrix.squeeze(), can_embedding.permute(0, 2, 1)).squeeze() # batch L T
            # t1 = self.W_a(t1.permute(0, 2, 1)).permute(0, 2, 1).squeeze() # batch  T
            # t1 = nn.Softmax(dim=1)(t1)

            # 2nd
            # X1 = news_matrix_1.squeeze() # news attention 的输出
            # X1 = news_matrix.squeeze() # trans的输出
            # can_id_embedding = self.item_embeddings(can_id_list)  # batch, 1000, 64
            # X1 = self.W_1(X1.permute(0, 2, 1)).permute(0, 2, 1) # batch 1, d
            # X2 = self.newsatt_net(can_embedding, can_ele_embedding, can_id_embedding, None, None, turn='candidate') # batch T d
            # t1 = torch.matmul(X1, X2.permute(0, 2, 1)).squeeze() # batch T
            # t1 = self.W_2(t1) # batch T
            # t1 = nn.Softmax(dim=-1)(t1)

            # 3rd: the candidates do not need to be similar to history, but to target(label)
            target_id_embedding = can_id_emb # batch, 64 can_id_emb
            can_id_embedding = self.item_embeddings(can_id_list)  # batch, 1000, 64
            X1 = self.newsatt_net(can_embed, can_element, target_id_embedding, None, None, turn='target').unsqueeze(1)  # batch 1 d target rep
            X2 = self.newsatt_net(can_embedding, can_ele_embedding, can_id_embedding, None, None, turn='candidate')  # batch T d
            t1 = torch.matmul(X1, X2.permute(0, 2, 1)).squeeze() # batch T
            t1 = self.W_x(t1)
            t1 = nn.Softmax(dim=-1)(t1)

            i1 = torch.argmax(t1, dim=-1) # batch, 1
            X2_t = X2[range(i1.size(0)), i1]
            loss = torch.mul(X1, X2_t) # batch
            loss = torch.sigmoid(loss)
            loss = torch.clamp(loss, min=1e-10, max=1-1e-10)
            loss = torch.log(loss)
            loss = -torch.mean(loss)

            values, indices = t1.topk(self.args.neg_samples, dim=1, largest=True, sorted=True)
            indices = torch.gather(can_id_list, dim=1, index=indices)

            # select negative samples
            return output, indices, loss
        else:
            return output