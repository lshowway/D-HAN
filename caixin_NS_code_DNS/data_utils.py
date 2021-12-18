# -*- coding: utf-8 -*-

import torch
import random
import numpy as np
import torch.nn.functional as F

activation_getter = {'iden': lambda x: x, 'relu': F.relu, 'tanh': F.tanh, 'sigm': F.sigmoid}


def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', 128)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)# 用的是这里

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def str2bool(v):
    return v.lower() in ('true')


# def generate_candidate(train):
#     candidate = dict()
#     all_items = np.arange(train.num_items - 1)
#     train_csr = train.tocsr()
#     for user, row in enumerate(train_csr):      # user是用户编号
#         candidate[user] = list(set(all_items) - set(row.indices))
#     print("load user candidate finished.")
#
#     return candidate

def generate_candidate(train):
    from dataset_DNS import N
    train_candidate = dict()
    test_candidate = dict()
    train_csr = train.tocsr()

    all_items = range(train.num_items - 1)
    for user, row in enumerate(train_csr):      # user是用户编号
        train_items = np.random.choice(a=all_items, size=5000, replace=False, p=None)  # sample 1W作为候选
        train_candidate[user] = list(set(train_items) - set(row.indices))
        test_candidate[user] = list(set(all_items) - set(row.indices))
    print("load user candidate finished.")

    return train_candidate, test_candidate