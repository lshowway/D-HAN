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


# def generate_candidate(file_path):
#     candidate = dict()
#     fin = open(file_path, 'r', encoding='UTF-8')
#     line = fin.readline().strip()
#     while line:
#         id = line
#         line = fin.readline().strip()
#         usercandnews = list(map(int, line.split(' ')))
#         candidate[int(id)] = usercandnews
#         line = fin.readline().strip()
#     print('candidate_news:'+str(len(candidate)))
#     return candidate


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
    print('candidate_news:'+str(len(candidate)))
    return train_candidate, test_candidate