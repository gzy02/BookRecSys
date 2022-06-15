import pandas as pd
import pickle
import os
import math
from operator import itemgetter
import heapq
import config
import torch
from NCFModel import NCFModel
from MFModel import MFModel

train = pd.read_csv('./datasets/train_dataset.csv')
data = train.copy()
data.pivot(index='user_id', columns='item_id')  # 这样会发现有大量的稀疏， 所以才会用字典进行存放
trainSet = {}
item_sim_matrix_path = "./pkl/matrix.pkl"
trainset_path = "./pkl/trainset.pkl"
item_sim_matrix_list_path = "./pkl/matrix_list_P_cov.pkl"


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


def generate_matrix_list(item_sim_matrix):
    for key in item_sim_matrix:
        test_list_key = sorted(item_sim_matrix[key].items(),
                               key=itemgetter(1),
                               reverse=True)
        item_sim_matrix[key] = test_list_key
    with open(item_sim_matrix_list_path, "wb") as fp:
        pickle.dump(item_sim_matrix, fp)


if not os.path.exists(trainset_path):
    Generate_trainSet()
if not os.path.exists(item_sim_matrix_path):
    Generate_matrix()
    #导入matrix
    with open(item_sim_matrix_path, "rb") as fp:
        item_sim_matrix = pickle.load(fp)
    generate_matrix_list(item_sim_matrix)

#导入trainset
with open(trainset_path, "rb") as fp:
    trainSet = pickle.load(fp)
with open(item_sim_matrix_list_path, "rb") as fp:
    item_sim_matrix = pickle.load(fp)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')
print(device)
traindataset_path = config.traindataset_path
with open(traindataset_path, "rb") as fp:
    traindataset = pickle.load(fp)

use_ncf_model = False
use_mf_model = True
if use_ncf_model:
    model = NCFModel(config.hidden_dim,
                     traindataset.user_nums,
                     traindataset.book_nums,
                     mlp_layer_num=config.mlp_layer_num,
                     dropout=0)  #dropout
    model_name = "ncf"
elif use_mf_model:
    model = MFModel(config.hidden_dim, traindataset.user_nums,
                    traindataset.book_nums)
    model_name = "mf"
else:
    model = None

BATCH_SIZE = 512


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


test = pd.read_csv(config.test_data_path)
user_lst = test['user_id'].tolist()

# 找到最相似的K个item， 最终推荐n个给用户
k = 2
n = 10


def main(it: int):

    model_path = config.model_path + str(it)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    with open(f'./submit/Item_CF_{model_name}_{it}_K={k}.csv', "w") as fp:
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

            if use_ncf_model or use_mf_model:  #使用模型进行精排
                user_input = torch.full(
                    [len(item_input_list)],
                    user_id).to(dtype=torch.int64).to(device)
                item_input = torch.Tensor(item_input_list).to(
                    dtype=torch.int64).to(device)
                result_output = model.my_predict(user_input, item_input).cpu()
            else:  #直接给一个单位向量，因为用户评分都默认为1
                result_output = [1 for i in range(len(item_input_list))]

            results = []
            for i in range(len(item_input_list)):
                results.append(
                    (item_input_list[i], item_list[item_input_list[i]] *
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


if __name__ == "__main__":
    main(300)