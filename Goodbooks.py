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
