#1、先把数据表给建立起
# 定义数据集， 也就是那个表格， 注意这里我们采用字典存放数据， 因为实际情况中数据是非常稀疏的， 很少有情况是现在这样
from email import header
import pickletools
import pandas as pd
#ABCDE是用户，其user_id分别是12345
#12345是物品


# def loadData():
#     items = {
#         'A': {
#             1: 5,
#             2: 3,
#             3: 4,
#             4: 3,
#             5: 1
#         },
#         'B': {
#             1: 3,
#             2: 1,
#             3: 3,
#             4: 3,
#             5: 5
#         },
#         'C': {
#             1: 4,
#             2: 2,
#             3: 4,
#             4: 1,
#             5: 5
#         },
#         'D': {
#             1: 4,
#             2: 3,
#             3: 3,
#             4: 5,
#             5: 2
#         },
#         'E': {
#             2: 3,
#             3: 5,
#             4: 4,
#             5: 1
#         }
#     }
#     users = {
#         1: {
#             'A': 5,
#             'B': 3,
#             'C': 4,
#             'D': 4
#         },
#         2: {
#             'A': 3,
#             'B': 1,
#             'C': 2,
#             'D': 3,
#             'E': 3
#         },
#         3: {
#             'A': 4,
#             'B': 3,
#             'C': 4,
#             'D': 3,
#             'E': 5
#         },
#         4: {
#             'A': 3,
#             'B': 3,
#             'C': 1,
#             'D': 5,
#             'E': 4
#         },
#         5: {
#             'A': 1,
#             'B': 5,
#             'C': 5,
#             'D': 2,
#             'E': 1
#         }
#     }
#     return items, users
import pickle
from typing import Dict,Set
import math
from operator import itemgetter
from collections import defaultdict
#from Utils import modelManager

class ItemCF(object):
    """ Item based Collaborative Filtering Algorithm Implementation"""
    def __init__(self, trainData, similarity="cosine", norm=True):
        self._trainData = trainData
        self._similarity = similarity
        self._isNorm = norm
        self._itemSimMatrix = dict() # 物品相似度矩阵

    def similarity(self):
        N = defaultdict(int) #记录每个物品的喜爱人数
        for user, items in self._trainData.items():
            for i in items:
                self._itemSimMatrix.setdefault(i, dict())
                N[i] += 1
                for j in items:
                    if i == j:
                        continue
                    self._itemSimMatrix[i].setdefault(j, 0)
                    if self._similarity == "cosine":
                        self._itemSimMatrix[i][j] += 1
                    elif self._similarity == "iuf":
                        self._itemSimMatrix[i][j] += 1. / math.log1p(len(items) * 1.)

        # print(self._itemSimMatrix)
        for i, related_items in self._itemSimMatrix.items():
            for j, cij in related_items.items():
                self._itemSimMatrix[i][j] = cij / math.sqrt(N[i]*N[j])
        # 是否要标准化物品相似度矩阵
        if self._isNorm:
            for i, relations in self._itemSimMatrix.items():
                max_num = relations[max(relations, key=relations.get)]
                # 对字典进行归一化操作之后返回新的字典
                self._itemSimMatrix[i] = {k : v/max_num for k, v in relations.items()}

    def recommend(self, user, N, K):
        """
        :param user: 被推荐的用户user
        :param N: 推荐的商品个数
        :param K: 查找的最相似的用户个数
        :return: 按照user对推荐物品的感兴趣程度排序的N个商品
        """
        recommends = dict()
        # 先获取user的喜爱物品列表
        items = self._trainData[user]

        for item in items:
            a = self._itemSimMatrix[item].items()
            # 对每个用户喜爱物品在物品相似矩阵中找到与其最相似的K个
            for i, sim in sorted(self._itemSimMatrix[item].items(), key=itemgetter(1), reverse=True)[:K]:
                if i in items:
                    continue  # 如果与user喜爱的物品重复了，则直接跳过
                recommends.setdefault(i, 0.)
                recommends[i] += sim
        # 根据被推荐物品的相似度逆序排列，然后推荐前N个物品给到用户
        return dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N])

    def train(self):
        try:
            print("start load item similarity matrix")
            with open("itemcf.pkl","rb") as f:
                self._itemSimMatrix = pickle.load(f)
        except BaseException as e:
            print("Exception occurs: " + str(e))
            print("load item similarity matrix failed, start train...")
            self.similarity()
            # save user similarity matrix
            with open("itemcf.pkl","wb") as f:
                pickle.dump(self._itemSimMatrix,f)


with open("items.pkl","rb") as f:
    user_items:Dict[int,Set[int]] = pickle.load(f)
with open("users.pkl","rb") as f:
    users:Dict[int,Dict[int,int]] = pickle.load(f)

print("data load data complete")
itemcf = ItemCF(user_items)
itemcf.train()
print("train finish")
fin = pd.DataFrame(columns=["user_id","item_id"])
num = 0
for i in users.keys():
    num+=1
    data = itemcf.recommend(i,10,100)
    for j in data.keys():
        fin = fin.append({'user_id':i, "item_id":j}, ignore_index=True)
    if not num%1000:
        print(num)
fin.to_csv("test_submission.csv")


# item_df = pd.DataFrame(items).T
# user_df = pd.DataFrame(users).T
# #2、计算用户相似性矩阵
# """计算用户相似性矩阵"""
# similarity_matrix = pd.DataFrame(np.zeros((len(items), len(items))),
#                                  index=list(items.keys()),
#                                  columns=list(items.keys()))
# # 遍历每条用户-物品评分数据
# for item_id in items:
#     for other_item_id in items:
#         vec_item = list()
#         vec_otheritem = list()
#         if item_id != other_item_id:
#             for itemId in items:  # 遍历物品-用户评分数据
#                 itemRatings = items[itemId]  # 这也是个字典  每条数据为所有用户对当前物品的评分
#                 if item_id in itemRatings and other_item_id in itemRatings:  # 说明两个用户都对该物品评过分
#                     vec_item.append(itemRatings[item_id])
#                     vec_otheritem.append(itemRatings[other_item_id])
#             # 这里可以获得相似性矩阵(共现矩阵)
#             #print(vec_user,vec_otheruser)
#             similarity_matrix[item_id][other_item_id] = np.corrcoef(
#                 np.array(vec_item), np.array(vec_otheritem))[0][1]
#             #similarity_matrix[item_id][other_item_id] = cosine_similarity(np.array(vec_user), np.array(vec_otheruser))[0][1]
#
# print(similarity_matrix)
# 3、计算前n个相似的用户
# """计算前n个相似的用户"""
# n = 2
# similarity_users = similarity_matrix[1].sort_values(
#     ascending=False)[:n].index.tolist()  # [2, 3]   也就是用户1和用户2
# 4、计算最终得分
# """计算最终得分"""
# base_score = np.mean(np.array([value for value in users[1].values()]))#用户1的均值
# weighted_scores = 0.
# corr_values_sum = 0.
# for user in similarity_users:  # [2, 3]
#     corr_value = similarity_matrix[1][user]  # 两个用户之间的相似性
#     mean_user_score = np.mean(
#         np.array([value for value in users[user].values()]))  # 每个用户的打分平均值
#     weighted_scores += corr_value * (users[user]['E'] - mean_user_score
#                                      )  # 加权分数
#     corr_values_sum += corr_value
# final_scores = base_score + weighted_scores / corr_values_sum
# print('用户Alice对物品5的打分预测: ', final_scores)
# user_df.loc[1]['E'] = final_scores
# print(user_df)