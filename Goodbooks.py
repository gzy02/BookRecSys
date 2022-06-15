# 1.数据准备
#1.1 数据说明
#1.2 数据预处理

#
# 1.1 加载数据
#观察样本数量和稀疏度。

import random
import tqdm
import torch
from torch.utils.data import Dataset
import pickle
import scipy.sparse as sp
from time import time
import numpy as np


class Goodbooks(Dataset):

    def __init__(self, df, user_book_map, mode='training', negs=99):
        super().__init__()

        self.df = df
        self.mode = mode
        self.user_book_map = user_book_map
        self.book_nums = max(df['item_id']) + 1
        self.user_nums = max(df['user_id']) + 1

        self._init_dataset()

    def _init_dataset(self):
        self.Xs = []

        if self.mode == 'training':
            for user, items in tqdm.tqdm(self.user_book_map.items()):
                for item in items[:-1]:
                    self.Xs.append((user, item, 1))  #默认评分是1

                    #或许不应该加入假的数据，没看过的不一定代表不喜欢
                    for _ in range(1):  # 1:1
                        while True:
                            neg_sample = random.randint(0, self.book_nums - 1)
                            #随机的label为假的数据
                            if neg_sample not in self.user_book_map[user]:
                                self.Xs.append((user, neg_sample, 0))  #设为0
                                break

        elif self.mode == 'validation':
            for user, items in tqdm.tqdm(self.user_book_map.items()):
                if len(items) == 0:
                    continue
                self.Xs.append((user, items[-1]))

    def __getitem__(self, index):

        if self.mode == 'training':
            user_id, book_id, label = self.Xs[index]
            return user_id, book_id, label
        elif self.mode == 'validation':  #没看过的99本随机加进去，与当前书一同作为验证集
            user_id, book_id = self.Xs[index]
            negs = list(
                random.sample(list(
                    set(range(self.book_nums)) -
                    set(self.user_book_map[user_id])),
                              k=99))
            return user_id, book_id, torch.LongTensor(negs)

    def __len__(self):
        return len(self.Xs)


def get_adj_mat(n_users, n_items, user_book_map):
    try:
        with open("./pkl/plain_adj.pkl", "rb") as fp:
            plain_adj = pickle.load(fp)
        with open("./pkl/norm_adj.pkl", "rb") as fp:
            norm_adj = pickle.load(fp)
        with open("./pkl/mean_adj.pkl", "rb") as fp:
            mean_adj = pickle.load(fp)
    except:
        plain_adj, norm_adj, mean_adj = generate_adj_mat(
            n_users, n_items, user_book_map)
        with open("./pkl/plain_adj.pkl", "wb") as fp:
            pickle.dump(plain_adj, fp)
        with open("./pkl/norm_adj.pkl", "wb") as fp:
            pickle.dump(norm_adj, fp)
        with open("./pkl/mean_adj.pkl", "wb") as fp:
            pickle.dump(mean_adj, fp)
    return plain_adj, norm_adj, mean_adj


class Goodbooks_GCN(Dataset):

    def __init__(self, df, user_book_map):
        super().__init__()

        self.df = df
        self.user_book_map = user_book_map
        self.book_nums = max(df['item_id']) + 1
        self.user_nums = max(df['user_id']) + 1

        self._init_dataset()

    def _init_dataset(self):
        self.Xs = []
        for user, items in tqdm.tqdm(self.user_book_map.items()):
            for item in items[:-1]:
                #加入假的数据
                while True:
                    neg_sample = random.randint(0, self.book_nums - 1)
                    #随机的label为假的数据
                    if neg_sample not in self.user_book_map[user]:
                        neg_item = neg_sample
                        break
                self.Xs.append((user, item, neg_item))

    def __getitem__(self, index):
        user_id, item, neg_item = self.Xs[index]
        return user_id, item, neg_item

    def __len__(self):
        return len(self.Xs)


def generate_adj_mat(n_users, n_items, user_book_map):
    t1 = time()
    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items),
                            dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    for user, items in user_book_map.items():
        for item in items:
            R[user, item] = 1.
    R = R.tolil()
    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok()
    print('already create adjacency matrix', adj_mat.shape, time() - t1)
    t2 = time()

    def mean_adj_single(adj):
        # D^-1 * A
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        # norm_adj = adj.dot(d_mat_inv)
        print('generate single-normalized adjacency matrix.')
        return norm_adj.tocoo()

    def normalized_adj_single(adj):
        # D^-1/2 * A * D^-1/2
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def check_adj_if_equal(adj):
        dense_A = np.array(adj.todense())
        degree = np.sum(dense_A, axis=1, keepdims=False)
        temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
        print(
            'check normalized adjacency matrix whether equal to this laplacian matrix.'
        )
        return temp

    norm_adj_mat = normalized_adj_single(adj_mat)
    # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
    mean_adj_mat_eye = mean_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
    print('already normalize adjacency matrix', time() - t2)
    return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat_eye.tocsr()