#5.模型预测
#加载测试数据
#生成提交文件

import pandas as pd
import pickle
import config
import torch
from LightGCN import LightGCN
import heapq
from Goodbooks import get_adj_mat

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')
print(device)
traindataset_path = "./pkl/GCN_TrainSet.pkl"
with open(traindataset_path, "rb") as fp:
    traindataset = pickle.load(fp)

if config.use_gcn:
    plain_adj, norm_adj, mean_adj = get_adj_mat(traindataset.user_nums,
                                                traindataset.book_nums,
                                                traindataset.user_book_map)
    model = LightGCN(traindataset.user_nums, traindataset.book_nums,
                     norm_adj).to(device)
    model_name = "gcn"
else:
    model = None

df = pd.read_csv(config.test_data_path)
user_for_test = df['user_id'].tolist()

BATCH_SIZE = 512


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main(it: int):

    model_path = config.model_path + str(it)
    submission_path = f"./submit/{model_name}_{it}.csv"
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    f = open(submission_path, 'w', encoding='utf-8')
    f.write("user_id,item_id\n")

    results = model.getUsersRating(user_for_test).cpu()
    for user_id in user_for_test:
        #将用户已经交互过的物品排除
        predict_item_id = []
        user_visited_items = traindataset.user_book_map[user_id]
        items_for_predict = list(
            set(range(traindataset.book_nums)) - set(user_visited_items))
        result_user = results[user_id].numpy().tolist()
        unsorted_list = list(enumerate(result_user))

        largest_210 = heapq.nlargest(
            210, unsorted_list,
            key=lambda x: x[1])  #因为看书最多的用户看了198本，找最大的210本足矣
        largest_210.sort(key=lambda x: x[1], reverse=True)
        cnt = 0
        for i in largest_210:
            if i[0] in items_for_predict:
                predict_item_id.append(i[0])
                cnt += 1
                if cnt == 10:
                    break
        tep_string = ""
        for x in predict_item_id:
            tep_string += '{},{}\n'.format(user_id, x)
        f.write(tep_string)
        if user_id % 1000 == 0:
            print(tep_string)
            f.flush()
    f.flush()
    f.close()


if __name__ == "__main__":
    main(1)