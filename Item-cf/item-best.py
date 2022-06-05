import random
import numpy as np
import pandas as pd
import pickle
import os
import math
from operator import itemgetter
import heapq

train = pd.read_csv('./datasets/train_dataset.csv')

data = train.copy()
data.pivot(index='user_id', columns='item_id')  # 这样会发现有大量的稀疏， 所以才会用字典进行存放
trainSet = {}
item_sim_matrix_path = "./matrix.pkl"
trainset_path = "./trainset.pkl"


def Generate_trainSet():
    for ele in data.itertuples():
        user, item = getattr(ele, 'user_id'), getattr(ele, 'item_id')
        trainSet.setdefault(user, {})
        trainSet[user][item] = 1  #默认用1来存放评分
    with open(trainset_path, "wb") as fp:
        pickle.dump(trainSet, fp)


def Generate_matrix():
    item_popular = {}
    for user, items in trainSet.items():
        for item in items:
            if item not in item_popular:
                item_popular[item] = 0
            item_popular[item] += 1

    item_count = len(item_popular)
    print('Total movie number = %d' % item_count)

    # 下面建立item相似矩阵
    print('Build user co-rated items matrix ...')
    item_sim_matrix = {}
    for user, items in trainSet.items():
        for m1 in items:  # 对于每个item， 都得双层遍历
            for m2 in items:
                if m1 == m2:
                    continue
                item_sim_matrix.setdefault(m1, {})
                item_sim_matrix[m1].setdefault(m2, 0)
                item_sim_matrix[m1][
                    m2] += 1  # 这里统计两个电影被同一个用户产生行为的次数， 这个就是余弦相似度的分子

    # 计算电影之间的相似性
    for m1, related_items in item_sim_matrix.items():
        for m2, count in related_items.items():
            # 这里item的用户数为0处理
            if item_popular[m1] == 0 or item_popular[m2] == 0:
                item_sim_matrix[m1][m2] = 0
            else:
                item_sim_matrix[m1][m2] = count / math.sqrt(
                    item_popular[m1] * item_popular[m2])

    with open(item_sim_matrix_path, "wb") as fp:
        pickle.dump(item_sim_matrix, fp)


if not os.path.exists(trainset_path):
    Generate_trainSet()
if not os.path.exists(item_sim_matrix_path):
    Generate_matrix()

#导入trainset
with open(trainset_path, "rb") as fp:
    trainSet = pickle.load(fp)

#导入matrix
with open(item_sim_matrix_path, "rb") as fp:
    item_sim_matrix = pickle.load(fp)

test = pd.read_csv('./datasets/test_dataset.csv')
sub = pd.read_csv('./datasets/submission.csv')

user_lst = test['user_id'].tolist()
#for user in user_lst:
#    if len(trainSet[user]) < 20:
#        print(user, len(trainSet[user]))

# 找到最相似的K个item， 最终推荐n个给用户
k = 2
n = 10

with open(f'./result/Item_CF_K={k}.csv', "w") as fp:
    fp.write("user_id,item_id\n")
    for user in user_lst:
        rank = {}
        watched_items = trainSet[user]  # 找出目标用户看过的书籍
        #if len(trainSet[user]) < 40:
        #    continue  #不管 看得少于40本的用户

        for item, rating in watched_items.items():
            #遍历与物品item最相似的前k个产品，获得这些物品及相似分数
            newdic = dict()  # 若该物品用户看过则不推荐
            for key, value in item_sim_matrix[item].items():
                if key not in watched_items:
                    newdic[key] = value  #去重

            predict_item_id = []
            largest_k = heapq.nlargest(k, newdic.items(), key=itemgetter(1))

            for related_item, w in largest_k:
                # 计算用户user对related_item的偏好值，初始化该值为0
                rank.setdefault(related_item, 0)
                #通过与其相似物品对物品related_item的偏好值相乘并相加。
                #排名的依据 : 推荐书籍与该已看书籍的相似度(累计) * 用户对已看书籍的评分
                rank[related_item] += w * float(rating)

        # 产生最后的推荐列表
        largest_n = heapq.nlargest(n, rank.items(), key=itemgetter(1))
        for i in largest_n:
            fp.write("{},{}\n".format(user, i[0]))
    fp.flush()
