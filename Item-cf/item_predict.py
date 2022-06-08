import pandas as pd
import pickle
import heapq

sim_list = ["E_dis", "P_cov", "J_sim", "cos"]
trainset_path = "../pkl/trainset.pkl"
item_sim_matrix_list_path = "../pkl/matrix_list_{}.pkl"

test = pd.read_csv('../datasets/test_dataset.csv')
user_lst = test['user_id'].tolist()

#导入trainset
with open(trainset_path, "rb") as fp:
    trainSet = pickle.load(fp)

# 对每个用户看过的item, 找到最相似的K个item, 最终推荐N个给用户
k = 5
n = 10

for name in sim_list:
    #读入相似度矩阵列表
    with open(item_sim_matrix_list_path.format(name), "rb") as fp:
        item_sim_matrix_list = pickle.load(fp)

    with open(f'../submit/Item_CF_{name}_K={k}.csv', "w") as fp:
        fp.write("user_id,item_id\n")
        for user_id in user_lst:
            item_list = {}  #待选的item集
            watched_items = trainSet[user_id]  # 找出目标用户看过的书籍
            #if len(trainSet[user]) < 40:
            #    continue  #不管 看得少于40本的用户

            for item, _ in watched_items.items():  #item是用户看过的物品之一
                #遍历与物品item最相似的前k个产品，获得这些物品及相似分数
                tep = 0
                for select_item in item_sim_matrix_list[item]:  #依次从排好序的列表里面取
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
            if user_id % 1000 == 0:
                print(tep_string)
                fp.flush()
        fp.flush()
        fp.close()
