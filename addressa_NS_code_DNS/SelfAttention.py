from transformers.modeling_bert import  ACT2FN
import torch
from torch import nn
import math

USE_CUDA = torch.cuda.is_available() and True
device = torch.device("cuda" if USE_CUDA else "cpu")


class ScaledDotSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.args = config
        self.attention_head_size = config.hidden_size
        self.all_head_size = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)

    def forward(self,hidden_states):
        query_layer = self.query(hidden_states)  # batch, L, dim
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # batch, L, L
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # batch, L, L
        # case study attention
        # if self.args.fw_sen is not None and self.args.theta < 1.0:
        #     batch = attention_probs.shape[0]
        #     for i in range(batch):
        #         t1 = attention_probs[i, :, :].tolist()
        #         self.args.fw_sen.write(str(t1) + '\n')
        #         self.args.fw_sen.flush()
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)  # batch, 1, len, len
        context_layer = torch.matmul(attention_probs, value_layer)  # batch, L, dim

        t1 = self.dense(context_layer)
        t1 = self.dropout(t1)
        attention_output = self.LayerNorm(t1 + hidden_states)  # batch, L, dim

        return attention_output


class MultiHeadAttention(nn.Module):
    """实现2017NIPS attention is all you need中的Figure 2 right"""
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)

    def transpose_for_scores(self, x):
        # 在这里将三维的batch,L,hidden_size变成多头的batch,L,num_attention_heads, attention_head_size
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # 1. 这个是Scaled Dot-Product Attention.
        mixed_query_layer = self.query(hidden_states)  # batch, L, dim
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # batch, L, L
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # batch, L, L
        # (left) Scaled Dot-Product Attention.中的mask操作，可选的，与softmax调换位置试试
        attention_probs = self.dropout(attention_probs)  # batch, 1, len, len
        context_layer = torch.matmul(attention_probs, value_layer)  # batch, L, hidden_size
        # 将多头之后的4D变成3d
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # BertSelfOutput：算上这个才是Multi-Head Attention，。
        t1 = self.dense(context_layer)
        t1 = self.dropout(t1)
        attention_output = self.LayerNorm(t1 + hidden_states)  # batch, L, dim

        return attention_output


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.args = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dense_2 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]
        self.dense_3 = nn.Linear(config.intermediate_size, config.hidden_size)


    def forward(self, hidden_states, attention_mask=None,):
        # 1.  multi head self-attention
        query_layer = self.query(hidden_states)  # batch, L, d
        key_layer = self.key(hidden_states) # batch, L, d
        value_layer = self.value(hidden_states) # batch, L, d

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # batch, L, L
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # batch, L, L

        # case study attention
        # if self.args.fw_sen is not None and self.args.theta < 1.0:
        #     batch = attention_probs.shape[0]
        #     for i in range(batch):
        #         t1 = attention_probs[i, :, :].tolist()
        #         self.args.fw_sen.write(str(t1) + '\n')
        #         self.args.fw_sen.flush()

        attention_probs = self.dropout(attention_probs)
        attn_vec = torch.matmul(attention_probs, value_layer)  #batch, L, d

        # 2. bertSelfOutput: drop 1 + add & Norm 1
        t1 = self.dense_1(attn_vec) # batch, L, d
        t1 = self.dropout(t1)
        attn_vec_2 = self.LayerNorm(t1 + hidden_states)  # batch, L, d 以上是scaled dot-product attention

        # 3.intermediate: Position-wise Feed-Forward
        attn_vec_3 = self.dense_2(attn_vec_2) # batch, L, I
        attn_vec_3 = self.intermediate_act_fn(attn_vec_3)

        # 4. BertOutput: + drop_2 + add & Norm 2
        attn_vec_3 = self.dense_3(attn_vec_3) # batch, L, d
        attn_vec_3 = self.dropout(attn_vec_3)
        attn_vec_4 = self.LayerNorm(attn_vec_3 + attn_vec_2)

        return attn_vec_4 # batch, L, d


class ScaleDotSelfAttentionAbs(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_head_size = config.hidden_size
        self.all_head_size = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)

    def forward(self, abs_encoding, hidden_states):
        query_layer = self.query(hidden_states)  # batch, L, dim
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        # QK
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # batch, L, L
        attention_scores_abs = torch.matmul(query_layer, abs_encoding.transpose(-1, -2))
        attention_scores = attention_scores + attention_scores_abs
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # batch, L, L
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)  # batch, 1, len, len

        # *V
        context_layer = torch.matmul(attention_probs, value_layer)  # batch, L, dim
        context_layer_abs = torch.matmul(attention_probs, abs_encoding)
        context_layer = context_layer + context_layer_abs

        t1 = self.dense(context_layer)
        t1 = self.dropout(t1)
        attention_output = self.LayerNorm(t1 + hidden_states)  # batch, L, dim
        return attention_output


class TransformerLayerAbs(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dense_2 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]
        self.dense_3 = nn.Linear(config.intermediate_size, config.hidden_size)


    def forward(self, abs_encoding, hidden_states, attention_mask=None,):
        # 1.  multi head self-attention
        query_layer = self.query(hidden_states)  # batch, L, d
        key_layer = self.key(hidden_states) # batch, L, d
        value_layer = self.value(hidden_states) # batch, L, d

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # batch, L, L
        attention_scores_abs = torch.matmul(query_layer, abs_encoding.transpose(-1, -2))
        attention_scores = attention_scores + attention_scores_abs
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # batch, L, L
        attention_probs = self.dropout(attention_probs)
        attn_vec = torch.matmul(attention_probs, value_layer)  #batch, L, d
        attn_vec_abs = torch.matmul(attention_probs, abs_encoding)
        attn_vec = attn_vec + attn_vec_abs

        # 2. bertSelfOutput: drop 1 + add & Norm 1
        t1 = self.dense_1(attn_vec) # batch, L, d
        t1 = self.dropout(t1)
        attn_vec_2 = self.LayerNorm(t1 + hidden_states)  # batch, L, d

        # 3.intermediate: Position-wise Feed-Forward
        attn_vec_3 = self.dense_2(attn_vec_2) # batch, L, I
        attn_vec_3 = self.intermediate_act_fn(attn_vec_3)

        # 4. BertOutput: + drop_2 + add & Norm 2
        attn_vec_3 = self.dense_3(attn_vec_3) # batch, L, d
        attn_vec_3 = self.dropout(attn_vec_3)
        attn_vec_4 = self.LayerNorm(attn_vec_3 + attn_vec_2)

        return attn_vec_4 # batch, L, d


class TransformerLayerWithMultiHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)

        self.dense_2 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]
        self.dense_3 = nn.Linear(config.intermediate_size, config.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # batch, L, head_num, head_size
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # batch, head_num, L, head_size

    def forward(self, hidden_states):
        # 1.  multi head self-attention
        mixed_query_layer = self.query(hidden_states)  # batch, L, d
        mixed_key_layer = self.key(hidden_states) # batch, L, d
        mixed_value_layer = self.value(hidden_states) # batch, L, d

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer) # batch, head_num, L, head_size

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # batch, head_num, L, L
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # batch, head_num L, L
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)  #batch, head_num, L, head_size

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # batch, L, head_num, head_size
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # batch, L, head_num*head_size
        context_layer = context_layer.view(*new_context_layer_shape) # batch, L, head_num*head_size

        # 2. bertSelfOutput: drop 1 + add & Norm 1
        t1 = self.dense_1(context_layer) # batch, L, d
        t1 = self.dropout(t1)
        attn_vec_2 = self.LayerNorm(t1 + hidden_states)  # batch, L, d

        # 3.intermediate: Position-wise Feed-Forward
        attn_vec_3 = self.dense_2(attn_vec_2) # batch, L, I
        attn_vec_3 = self.intermediate_act_fn(attn_vec_3)

        # 4. BertOutput: + drop_2 + add & Norm 2
        attn_vec_3 = self.dense_3(attn_vec_3) # batch, L, d
        attn_vec_3 = self.dropout(attn_vec_3)
        attn_vec_4 = self.LayerNorm(attn_vec_3 + attn_vec_2)

        return attn_vec_4 # batch, L, d


class TimeEmbeddingMonth(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.args = config
        t = config.hidden_size // 5
        self.m_embedding = nn.Embedding(12, t)
        self.d_embedding = nn.Embedding(31, t)
        self.w_embedding = nn.Embedding(7, t)
        self.h_embedding = nn.Embedding(24, t)
        self.mi_embedding = nn.Embedding(60, config.hidden_size - t * 4)

        self.m_interval_embedding = nn.Embedding(12 * 2, t)
        self.d_interval_embedding = nn.Embedding(31 * 2, t)
        self.w_interval_embedding = nn.Embedding(7 * 2, t) # -7, 6
        self.h_interval_embedding = nn.Embedding(24 * 2, t) # -24, 24
        self.mi_interval_embedding = nn.Embedding(60 * 2, config.hidden_size - t * 4) # -60, 60

    def convert_timestamp(self, history_time, candidate_time):
        import time
        batch_time = torch.cat([history_time, candidate_time.unsqueeze(1)], dim=1).tolist()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        months = dict(zip(months, range(12)))

        days = dict(zip(range(1, 31+1), range(31)))

        weeks = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weeks = dict(zip(weeks, range(7)))

        hours = dict(zip(range(24), range(24)))
        miniutes = dict(zip(range(60), range(60)))

        def convert(timestamp):
            t = time.ctime(timestamp)
            # print(t)
            w, m, d,hm, y = t.split()
            h, mi = hm.split(':')[0:2]
            return [y, m, d, w, h, mi]
        batch_m, batch_d, batch_w, batch_h, batch_mi = [], [], [], [], []
        for time_list in batch_time:
            t = [convert(x) for x in time_list]
            m_list, d_list, w_list, h_list, mi_list = [], [], [], [], []
            for x in t:
                y, m, d, w, h, mi = x
                d, h, mi = int(d), int(h), int(mi)
                m, d, w, h, mi = months[m], days[d], weeks[w], hours[h], miniutes[mi]
                m_list.append(m)
                d_list.append(d)
                w_list.append(w)
                h_list.append(h)
                mi_list.append(mi)
            batch_m.append(m_list)
            batch_d.append(d_list)
            batch_w.append(w_list)
            batch_h.append(h_list)
            batch_mi.append(mi_list)
        return batch_m, batch_d, batch_w, batch_h, batch_mi # batch, L

    def date_to_tensor(self, history_time, candidate_time):
        batch_m, batch_d, batch_w, batch_h, batch_mi = self.convert_timestamp(history_time, candidate_time)
        batch_m = torch.tensor(batch_m).to(device)
        batch_d = torch.tensor(batch_d).to(device)
        batch_w = torch.tensor(batch_w).to(device)
        batch_h = torch.tensor(batch_h).to(device)
        batch_mi = torch.tensor(batch_mi).to(device)

        return  batch_m, batch_d, batch_w, batch_h, batch_mi

    def get_absolute_embedding(self, batch_m, batch_d, batch_w, batch_h, batch_mi):
        m_embed = self.m_embedding(batch_m)
        d_embed = self.d_embedding(batch_d)
        w_embed = self.w_embedding(batch_w)
        h_embed = self.h_embedding(batch_h)
        mi_embed = self.mi_embedding(batch_mi)

        abs_embedding = torch.cat([m_embed, d_embed, w_embed, h_embed, mi_embed], dim=2)

        return abs_embedding

    def get_interval_embedding(self, batch_m, batch_d, batch_w, batch_h, batch_mi):
        m_interval = batch_m[:, -1].unsqueeze(1) - batch_m[:, :-1] + 12 # candidate - history
        d_interval = batch_d[:, -1].unsqueeze(1) - batch_d[:, :-1] + 31
        w_interval = batch_w[:, -1].unsqueeze(1) - batch_w[:, :-1] + 7
        h_interval = batch_h[:, -1].unsqueeze(1) - batch_h[:, :-1] + 24
        mi_interval = batch_mi[:, -1].unsqueeze(1) - batch_mi[:, :-1] + 60

        m_interval_embed = self.m_interval_embedding(m_interval)
        d_interval_embed = self.d_interval_embedding(d_interval)
        w_interval_embed = self.w_interval_embedding(w_interval)
        h_interval_embed = self.h_interval_embedding(h_interval)
        mi_interval_embed = self.mi_interval_embedding(mi_interval)

        interval_embedding = torch.cat([m_interval_embed, d_interval_embed, w_interval_embed, h_interval_embed, mi_interval_embed], dim=2)

        return interval_embedding

    def forward(self, history_time, candidate_time):
        batch_m, batch_d, batch_w, batch_h, batch_mi = self.date_to_tensor(history_time, candidate_time)
        abs_embedding = self.get_absolute_embedding(batch_m, batch_d, batch_w, batch_h, batch_mi)
        interval_embedding = self.get_interval_embedding(batch_m, batch_d, batch_w, batch_h, batch_mi)

        return abs_embedding, interval_embedding


class TimeEmbeddingAdd(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.args = config
        t = config.hidden_size
        self.m_embedding = nn.Embedding(12, t)
        self.d_embedding = nn.Embedding(31, t)
        self.w_embedding = nn.Embedding(7, t)
        self.h_embedding = nn.Embedding(24, t)
        self.mi_embedding = nn.Embedding(60, t)

        self.m_interval_embedding = nn.Embedding(12 * 2, t)
        self.d_interval_embedding = nn.Embedding(31 * 2, t)
        self.w_interval_embedding = nn.Embedding(7 * 2, t) # -7, 6
        self.h_interval_embedding = nn.Embedding(24 * 2, t) # -24, 24
        self.mi_interval_embedding = nn.Embedding(60 * 2, t) # -60, 60

    def convert_timestamp(self, history_time, candidate_time):
        import time
        batch_time = torch.cat([history_time, candidate_time.unsqueeze(1)], dim=1).tolist()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        months = dict(zip(months, range(12)))

        days = dict(zip(range(1, 31+1), range(31)))

        weeks = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weeks = dict(zip(weeks, range(7)))

        hours = dict(zip(range(24), range(24)))
        miniutes = dict(zip(range(60), range(60)))

        def convert(timestamp):
            t = time.ctime(timestamp)
            # print(t)
            w, m, d,hm, y = t.split()
            h, mi = hm.split(':')[0:2]
            return [y, m, d, w, h, mi]
        batch_m, batch_d, batch_w, batch_h, batch_mi = [], [], [], [], []
        for time_list in batch_time:
            t = [convert(x) for x in time_list]
            m_list, d_list, w_list, h_list, mi_list = [], [], [], [], []
            for x in t:
                y, m, d, w, h, mi = x
                d, h, mi = int(d), int(h), int(mi)
                m, d, w, h, mi = months[m], days[d], weeks[w], hours[h], miniutes[mi]
                m_list.append(m)
                d_list.append(d)
                w_list.append(w)
                h_list.append(h)
                mi_list.append(mi)
            batch_m.append(m_list)
            batch_d.append(d_list)
            batch_w.append(w_list)
            batch_h.append(h_list)
            batch_mi.append(mi_list)
        return batch_m, batch_d, batch_w, batch_h, batch_mi # batch, L

    def date_to_tensor(self, history_time, candidate_time):
        batch_m, batch_d, batch_w, batch_h, batch_mi = self.convert_timestamp(history_time, candidate_time)
        batch_m = torch.tensor(batch_m).to(device)
        batch_d = torch.tensor(batch_d).to(device)
        batch_w = torch.tensor(batch_w).to(device)
        batch_h = torch.tensor(batch_h).to(device)
        batch_mi = torch.tensor(batch_mi).to(device)

        return  batch_m, batch_d, batch_w, batch_h, batch_mi

    def get_absolute_embedding(self, batch_m, batch_d, batch_w, batch_h, batch_mi):
        m_embed = self.m_embedding(batch_m)
        d_embed = self.d_embedding(batch_d)
        w_embed = self.w_embedding(batch_w)
        h_embed = self.h_embedding(batch_h)
        mi_embed = self.mi_embedding(batch_mi) # batch, L, d

        abs_embedding = torch.stack([m_embed, d_embed, w_embed, h_embed, mi_embed], dim=2)
        abs_embedding = torch.mean(abs_embedding, dim=2)

        return abs_embedding

    def get_interval_embedding(self, batch_m, batch_d, batch_w, batch_h, batch_mi):
        m_interval = batch_m[:, -1].unsqueeze(1) - batch_m[:, :-1] + 12 # candidate - history
        d_interval = batch_d[:, -1].unsqueeze(1) - batch_d[:, :-1] + 31
        w_interval = batch_w[:, -1].unsqueeze(1) - batch_w[:, :-1] + 7
        h_interval = batch_h[:, -1].unsqueeze(1) - batch_h[:, :-1] + 24
        mi_interval = batch_mi[:, -1].unsqueeze(1) - batch_mi[:, :-1] + 60

        m_interval_embed = self.m_interval_embedding(m_interval)
        d_interval_embed = self.d_interval_embedding(d_interval)
        w_interval_embed = self.w_interval_embedding(w_interval)
        h_interval_embed = self.h_interval_embedding(h_interval)
        mi_interval_embed = self.mi_interval_embedding(mi_interval)

        interval_embedding = torch.stack([m_interval_embed, d_interval_embed, w_interval_embed, h_interval_embed, mi_interval_embed], dim=2)
        interval_embedding = torch.mean(interval_embedding, dim=2)

        return interval_embedding

    def forward(self, history_time, candidate_time):
        batch_m, batch_d, batch_w, batch_h, batch_mi = self.date_to_tensor(history_time, candidate_time)
        abs_embedding = self.get_absolute_embedding(batch_m, batch_d, batch_w, batch_h, batch_mi)
        interval_embedding = self.get_interval_embedding(batch_m, batch_d, batch_w, batch_h, batch_mi)

        return abs_embedding, interval_embedding