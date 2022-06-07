import random
import numpy as np
import pandas as pd
import pickle
import os
import math
from operator import itemgetter
import heapq

train = pd.read_csv('../datasets/train_dataset.csv')

data = train.copy()
data.pivot(index='user_id', columns='item_id')  # 这样会发现有大量的稀疏， 所以才会用字典进行存放
trainSet = {}
sim_list = ["E_dis","P_cov","J_sim"]
item_sim_matrix_path = "./pkl/matrix{}.pkl"
trainset_path = "./pkl/trainset.pkl"
item_sim_matrix_list_path = "./pkl/matrix_list{}.pkl"

def similar(count:int,popular_A:int,popular_B:int,length:int,para:str)->float:
    if para == "E_dis":
        result = 1/math.sqrt(popular_A+popular_B-2*count)
    elif para == "P_cov":
        result = (length*count-popular_A*popular_B)/\
            math.sqrt(popular_A*popular_B*(length-popular_A)*(length-popular_B))
    elif para == "J_sim":
        result = count/(popular_A+popular_B-count)
    else:
        result = count/math.sqrt(popular_A*popular_B)
    return result

def Generate_trainSet():
    for ele in data.itertuples():
        user, item = getattr(ele, 'user_id'), getattr(ele, 'item_id')
        trainSet.setdefault(user, {})
        trainSet[user][item] = 1  #默认用1来存放评分
    with open(trainset_path, "wb") as fp:
        pickle.dump(trainSet, fp)


def Generate_matrix(para:str):
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
                item_sim_matrix[m1][m2] = similar(
                count,item_popular[m1],item_popular[m2],len(trainSet),para=para)
                #count / math.sqrt(
                    #item_popular[m1] * item_popular[m2])

    with open(item_sim_matrix_path, "wb") as fp:
        pickle.dump(item_sim_matrix, fp)


def generate_matrix_list(item_sim_matrix,para:str):
    for key in item_sim_matrix:
        test_list_key = sorted(item_sim_matrix[key].items(),
                               key=itemgetter(1),
                               reverse=True)
        item_sim_matrix[key] = test_list_key
    with open(item_sim_matrix_list_path.format("_"+para), "wb") as fp:
        pickle.dump(item_sim_matrix, fp)


if not os.path.exists(trainset_path):
    Generate_trainSet()
for i in sim_list:
    if not os.path.exists(item_sim_matrix_path.format("_"+i)):
        Generate_matrix(i)
        #导入matrix
        with open(item_sim_matrix_path.format("_"+i), "rb") as fp:
            item_sim_matrix = pickle.load(fp)
        generate_matrix_list(item_sim_matrix,i)

#导入trainset
with open(trainset_path, "rb") as fp:
    trainSet = pickle.load(fp)
with open(item_sim_matrix_list_path.format(""), "rb") as fp:
    item_sim_matrix = pickle.load(fp)

test = pd.read_csv('./datasets/test_dataset.csv')
user_lst = test['user_id'].tolist()

# 找到最相似的K个item， 最终推荐n个给用户
k = 2
n = 10

with open(f'./submit/Item_CF_K={k}_2.csv', "w") as fp:
    fp.write("user_id,item_id\n")
    for user_id in user_lst:
        item_list = {}  #待选的item集
        watched_items = trainSet[user_id]  # 找出目标用户看过的书籍
        #if len(trainSet[user]) < 40:
        #    continue  #不管 看得少于40本的用户

        for item, _ in watched_items.items():  #item是用户看过的物品之一
            #遍历与物品item最相似的前k个产品，获得这些物品及相似分数
            tep = 0
            for select_item in item_sim_matrix[item]:  #依次从排好序的列表里面取
                if select_item[0] not in watched_items:  #没看过
                    # 计算用户user对select_item的偏好值，初始化该值为0
                    item_list.setdefault(select_item[0], 0)
                    #通过与其相似物品对物品select_item的偏好值相乘并相加。
                    #排名的依据 : 推荐书籍与该已看书籍的相似度(累计) * NCF模型对该书籍的评分
                    item_list[select_item[0]] += select_item[1]
                    tep += 1
                    if tep == k:
                        break
        item_input_list = list(item_list.keys())

        result_output = [1 for i in range(len(item_input_list))]
        results = []
        for i in range(len(item_input_list)):
            results.append((item_input_list[i], item_list[item_input_list[i]] *
                            result_output[i]))  #item_id,item_score

        largest_10 = heapq.nlargest(10, results, key=lambda x: x[1])
        predict_item_id = []
        for i in largest_10:
            predict_item_id.append(i[0])

        tep_string = ""
        for x in predict_item_id:
            tep_string += '{},{}\n'.format(user_id, x)
        fp.write(tep_string)
        if user_id % 100 == 0:
            print(tep_string)
            fp.flush()
    fp.flush()
    fp.close()
