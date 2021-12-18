# -*- coding: utf-8 -*-

import time
import numpy as np
import scipy.sparse as sp

def _sliding_window(tensor, tensor_time, window_size, step_size=1):
    if len(tensor) - window_size >= 0:  # 用户的阅读序列超过L+T
        for i in range(len(tensor), 0, -step_size):
            if i - window_size >= 0:
                yield tensor[i - window_size:i],tensor_time[i - window_size:i]
            else:
                break
    else:                               # 用户的阅读序列小于L+T
        yield tensor, tensor_time


def _generate_sequences(user_ids, item_ids, timestamps,
                        indices,
                        max_sequence_length):
    for i in range(len(indices)): # 循环每个用户

        # start_idx:一个用户开始的位置; stop_idx:一个用户结束的位置
        start_idx = indices[i]
        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]

        for (seq, seq_time) in _sliding_window(item_ids[start_idx:stop_idx],timestamps[start_idx:stop_idx],
                                   max_sequence_length):
            yield (user_ids[i], seq, seq_time) # seq是一个序列


class Interactions(object):

    def __init__(self, file_path,
                 generate_seq_f,
                 user_map=None,
                 item_map=None):

        if not user_map and not item_map:
            user_map = dict()
            item_map = dict()

            num_user = 0
            num_item = 0
        else:
            num_user = len(user_map)
            num_item = len(item_map)

        # userId、newsId、timestamps list
        user_ids = list()
        item_ids = list()
        timestamps = list()
        with open(file_path, 'r') as fin:
            if 'caixin' in file_path:
                print("Interactions 处理时间用的 caixin 标准。")
                for line in fin:
                    u, i, t = line.strip().split('\t') # 用户 新闻 时间
                    user_ids.append(u)
                    item_ids.append(i)
                    timestamps.append(int(t)) # str-->timestamp(10位，int32就够)
            elif 'addressa' in file_path:
                print("Interactions 处理时间用的 addressa 标准。")
                for line in fin:
                    u, i, t = line.strip().split('\t')
                    user_ids.append(u)
                    item_ids.append(i)
                    timestamps.append(int(t)) # str-->timestamp(10位，int32就够)
            else:
                print("Interactions 处理时间用的 cert 标准。")
                for line in fin:
                    u, i, t = line.strip().split('\t')
                    user_ids.append(u)
                    item_ids.append(i)
                    timestamps.append(int(time.mktime(time.strptime(t.strip(), "%Y-%m-%d %H:%M:%S"))))  # str-->timestamp(10位，int32就够)


        # user_map[userId - user编号(从0开始)] str - int
        # item_map[newsId - news编号(从0开始)] str - int
        for u in user_ids:
            if u not in user_map:
                user_map[u] = num_user
                num_user += 1
        for i in item_ids:
            if i not in item_map:
                item_map[i] = num_item
                num_item += 1


        # user编号、news编号 list(都从0开始)
        user_ids = np.array([user_map[u] for u in user_ids],dtype = np.int64)
        item_ids = np.array([item_map[i] for i in item_ids],dtype = np.int64)
        timestamps = np.array(timestamps)

        self.num_users = num_user
        self.num_items = num_item

        self.user_ids = user_ids
        self.item_ids = item_ids
        self.timestamps = timestamps

        self.user_map = user_map
        self.item_map = item_map

        self.generate_seq_f = generate_seq_f

        self.sequences = None
        self.test_sequences = None

    def __len__(self):

        return len(self.user_ids)

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.user_ids
        col = self.item_ids
        data = np.ones(len(self))

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()

    def to_sequence(self, sequence_length=5):
        """
        Transform to sequence form.Valid subsequences of users' interactions are returned. For example, if a user interacted with items
        [1, 2, 3, 4, 5, 6, 7, 8, 9], the returned interactions matrix at sequence length 5 and target length 3 will be be given by:

        sequences:
           [[1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]]
        targets:
           [[6, 7],
            [7, 8],
            [8, 9]]
        sequence for test (the last 'sequence_length' items of each user's sequence):
        [[5, 6, 7, 8, 9]]

        Parameters:
        sequence_length: int
            Sequence length. Subsequences shorter than this will be left-padded with zeros.
        target_length: int
            Sequence target length.
        """

        target_length = 1
        max_sequence_length = sequence_length + target_length

        # Sort first by user id, then by timestamp
        sort_indices = np.lexsort((self.timestamps,
                                   self.user_ids))

        user_ids = self.user_ids[sort_indices]
        item_ids = self.item_ids[sort_indices]
        timestamps = self.timestamps[sort_indices]

        user_ids, indices, counts = np.unique(user_ids,
                                              return_index=True,
                                              return_counts=True)

        num_subsequences = sum([c - max_sequence_length + 1 if c >= max_sequence_length else 1 for c in counts])#一共有多少子序列。1573163 = 1573760-199 * (4-1)

        sequences = np.zeros((num_subsequences, sequence_length),
                             dtype=np.int64)
        sequences_time = np.zeros((num_subsequences, sequence_length),
                                  dtype=np.int64)
        sequences_targets = np.zeros(num_subsequences,
                                     dtype=np.int64)
        sequences_targets_time = np.zeros(num_subsequences,
                                     dtype=np.int64)
        sequence_users = np.empty(num_subsequences,
                                  dtype=np.int64)

        test_sequences = np.zeros((self.num_users, sequence_length),
                                  dtype=np.int64)
        test_sequences_time = np.zeros((self.num_users, sequence_length),
                                  dtype=np.int64)
        test_users = np.empty(self.num_users,
                              dtype=np.int64)

        _uid = None
        for i, (uid,item_seq,time_seq) in enumerate(_generate_sequences(user_ids,# uid,item_seq表示uid与他的一个序列
                                                           item_ids,timestamps,
                                                           indices,
                                                           max_sequence_length)):
            if uid != _uid:# 到了一个新的用户，该用户最后交互的L个item是test数据（item_seq先返回后读的）
                test_sequences[uid][:] = item_seq[-sequence_length:]
                test_sequences_time[uid][:] = time_seq[-sequence_length:]
                test_users[uid] = uid
                _uid = uid
            sequences_targets[i] = item_seq[-target_length]         # 后1个item
            sequences_targets_time[i] = time_seq[-target_length]
            sequences[i][:] = item_seq[:sequence_length]            # 前L个item
            sequences_time[i][:] = time_seq[:sequence_length]
            sequence_users[i] = uid
        self.sequences = SequenceInteractions(sequence_users, sequences, sequences_time, sequences_targets, sequences_targets_time)
        self.test_sequences = SequenceInteractions(test_users, test_sequences, test_sequences_time)

class SequenceInteractions(object):
    """
    Interactions encoded as a sequence matrix.

    Parameters
    ----------
    user_ids: np.array
        sequence users
    sequences: np.array
        The interactions sequence matrix, as produced by
        :func:`~Interactions.to_sequence`
    targets: np.array
        sequence targets
    """

    def __init__(self,
                 user_ids,
                 sequences,
                 sequences_time,
                 targets=None,
                 targets_time=None):
        self.user_ids = user_ids
        self.sequences = sequences
        self.sequences_time = sequences_time
        self.targets = targets
        self.targets_time = targets_time

        self.L = sequences.shape[1]
