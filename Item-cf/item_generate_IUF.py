import pandas as pd
import pickle
import os
import math
from operator import itemgetter
import copy

data = pd.read_csv('../datasets/train_dataset.csv')
data.pivot(index='user_id', columns='item_id')  # 这样会发现有大量的稀疏， 所以才会用字典进行存放
trainSet = {}
item_popular = {}
item_sim_matrix_count = {}

sim_list = ["IUF"]
item_sim_matrix_path = "../pkl/matrix{}.pkl"
item_sim_matrix_list_path = "../pkl/matrix_list{}.pkl"

trainset_path = "../pkl/trainset.pkl"
item_popular_path = "../pkl/item_popular.pkl"
item_sim_matrix_count_path = "../pkl/matrix_count_IUF.pkl"


def similar(count: int, popular_A: int, popular_B: int) -> float:
    result = count / math.sqrt(popular_A * popular_B)
    return result


def Generate_trainSet():
    for ele in data.itertuples():
        user, item = getattr(ele, 'user_id'), getattr(ele, 'item_id')
        trainSet.setdefault(user, {})
        trainSet[user][item] = 1  #默认用1来存放评分
    with open(trainset_path, "wb") as fp:
        pickle.dump(trainSet, fp)


def Generate_item_popular():
    for user, items in trainSet.items():
        for item in items:
            if item not in item_popular:
                item_popular[item] = 0
            item_popular[item] += 1

    with open(item_popular_path, "wb") as fp:
        pickle.dump(item_popular, fp)


def Generate_matrix_count():
    for user, items in trainSet.items():
        for m1 in items:  # 对于每个item， 都得双层遍历
            for m2 in items:
                if m1 == m2:
                    continue
                item_sim_matrix_count.setdefault(m1, {})
                item_sim_matrix_count[m1].setdefault(m2, 0)
                #print(trainSet[user])
                #print(len(trainSet[user]))
                item_sim_matrix_count[m1][m2] += 1 / math.log2(
                    1 + len(trainSet[user]))
                # 这里统计两个item被同一个用户产生行为的次数， 这个就是IUF的分子
    with open(item_sim_matrix_count_path, "wb") as fp:
        pickle.dump(item_sim_matrix_count, fp)


def Generate_matrix(para: str):
    # 下面建立item相似矩阵
    print(f'Build user co-rated items matrix_{para} ...')
    item_sim_matrix = copy.deepcopy(item_sim_matrix_count)

    # 计算item之间的相似性
    for m1, related_items in item_sim_matrix.items():
        for m2, count in related_items.items():
            # 这里item的用户数为0处理
            if item_popular[m1] == 0 or item_popular[m2] == 0:
                item_sim_matrix[m1][m2] = 0
            else:
                item_sim_matrix[m1][m2] = similar(count, item_popular[m1],
                                                  item_popular[m2])
    with open(item_sim_matrix_path.format("_" + para), "wb") as fp:
        pickle.dump(item_sim_matrix, fp)


def generate_matrix_list(item_sim_matrix, para: str):
    for key in item_sim_matrix:
        test_list_key = sorted(item_sim_matrix[key].items(),
                               key=itemgetter(1),
                               reverse=True)
        item_sim_matrix[key] = test_list_key
    with open(item_sim_matrix_list_path.format("_" + para), "wb") as fp:
        pickle.dump(item_sim_matrix, fp)


#导入trainset
if not os.path.exists(trainset_path):
    Generate_trainSet()
with open(trainset_path, "rb") as fp:
    trainSet = pickle.load(fp)

#导入item_popular
if not os.path.exists(item_popular_path):
    Generate_item_popular()
with open(item_popular_path, "rb") as fp:
    item_popular = pickle.load(fp)

#打印user,item的count
user_count = len(trainSet)
item_count = len(item_popular)
print('Total item number = %d\nTotal user number = %d' %
      (item_count, user_count))

#导入sim_matrix_count
if not os.path.exists(item_sim_matrix_count_path):
    Generate_matrix_count()
with open(item_sim_matrix_count_path, "rb") as fp:
    item_sim_matrix_count = pickle.load(fp)

#生成相似度矩阵
for i in sim_list:
    if not os.path.exists(item_sim_matrix_path.format("_" + i)):
        Generate_matrix(i)
        #导入matrix
        with open(item_sim_matrix_path.format("_" + i), "rb") as fp:
            item_sim_matrix = pickle.load(fp)
        generate_matrix_list(item_sim_matrix, i)
