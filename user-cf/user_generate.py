import pandas as pd
import pickle
import os
import math
from operator import itemgetter
import copy

data = pd.read_csv('../datasets/train_dataset.csv')
data.pivot(index='user_id', columns='item_id')  # 这样会发现有大量的稀疏， 所以才会用字典进行存放
trainSet = {}
user_popular = {}
user_sim_matrix_count = {}

sim_list = ["E_dis", "P_cov", "J_sim"]
user_sim_matrix_path = "../pkl/user_matrix{}.pkl"
user_sim_matrix_list_path = "../pkl/user_matrix_list{}.pkl"

trainset_path = "../pkl/user_trainset.pkl"
user_popular_path = "../pkl/user_popular.pkl"
user_sim_matrix_count_path = "../pkl/user_matrix_count.pkl"


#用户A看过的书的数量 用户B看过的书的数量 他们共同看过的书的数量
def similar(count: int, popular_A: int, popular_B: int, length: int,
            para: str) -> float:
    if para == "E_dis":
        result = 1 / (1 + math.sqrt(popular_A + popular_B - 2 * count))
    elif para == "P_cov":
        result = (length*count-popular_A*popular_B)/\
            math.sqrt(popular_A*popular_B*(length-popular_A)*(length-popular_B))
    elif para == "J_sim":
        result = count / (popular_A + popular_B - count)
    else:
        result = count / math.sqrt(popular_A * popular_B)
    return result


def Generate_trainSet():
    for ele in data.itertuples():
        user, item = getattr(ele, 'user_id'), getattr(ele, 'item_id')
        trainSet.setdefault(item, {})
        trainSet[item][user] = 1  #默认用1来存放评分
    with open(trainset_path, "wb") as fp:
        pickle.dump(trainSet, fp)


def Generate_user_popular():
    for items, users in trainSet.items():
        for user in users:
            if user not in user_popular:
                user_popular[user] = 0
            user_popular[user] += 1

    with open(user_popular_path, "wb") as fp:
        pickle.dump(user_popular, fp)


def Generate_matrix_count():
    for items, users in trainSet.items():
        for m1 in users:  # 对于每个user， 都得双层遍历
            for m2 in users:
                if m1 == m2:
                    continue
                user_sim_matrix_count.setdefault(m1, {})
                user_sim_matrix_count[m1].setdefault(m2, 0)
                user_sim_matrix_count[m1][m2] += 1
                # 这里统计两个user都对同一个item产生行为的次数，即count
    with open(user_sim_matrix_count_path, "wb") as fp:
        pickle.dump(user_sim_matrix_count, fp)


def Generate_matrix(para: str):
    # 下面建立user相似矩阵
    print(f'Build item co-rated users matrix_{para} ...')
    user_sim_matrix = copy.deepcopy(user_sim_matrix_count)

    # 计算user之间的相似性
    for m1, related_items in user_sim_matrix.items():
        for m2, count in related_items.items():
            # 一本书都没看过，直接0
            if user_popular[m1] == 0 or user_popular[m2] == 0:
                user_sim_matrix[m1][m2] = 0
            else:
                user_sim_matrix[m1][m2] = similar(count,
                                                  user_popular[m1],
                                                  user_popular[m2],
                                                  len(trainSet),
                                                  para=para)
    with open(user_sim_matrix_path.format("_" + para), "wb") as fp:
        pickle.dump(user_sim_matrix, fp)


def generate_matrix_list(user_sim_matrix, para: str):
    for key in user_sim_matrix:
        test_list_key = sorted(user_sim_matrix[key].items(),
                               key=itemgetter(1),
                               reverse=True)
        user_sim_matrix[key] = test_list_key
    with open(user_sim_matrix_list_path.format("_" + para), "wb") as fp:
        pickle.dump(user_sim_matrix, fp)


#导入trainset
if not os.path.exists(trainset_path):
    Generate_trainSet()
with open(trainset_path, "rb") as fp:
    trainSet = pickle.load(fp)

#导入user_popular
if not os.path.exists(user_popular_path):
    Generate_user_popular()
with open(user_popular_path, "rb") as fp:
    user_popular = pickle.load(fp)

#打印user,item的count
item_count = len(trainSet)
user_count = len(user_popular)
print('Total item number = %d\nTotal user number = %d' %
      (item_count, user_count))

#导入sim_matrix_count
if not os.path.exists(user_sim_matrix_count_path):
    Generate_matrix_count()
with open(user_sim_matrix_count_path, "rb") as fp:
    user_sim_matrix_count = pickle.load(fp)

#生成相似度矩阵
for i in sim_list:
    if not os.path.exists(user_sim_matrix_path.format("_" + i)):
        Generate_matrix(i)
        #导入matrix
        with open(user_sim_matrix_path.format("_" + i), "rb") as fp:
            user_sim_matrix = pickle.load(fp)
        generate_matrix_list(user_sim_matrix, i)
