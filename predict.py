#5.模型预测
#加载测试数据
#生成提交文件

import numpy as np
import pandas as pd
import pickle
import config
import os
import torch

from NCFModel import NCFModel

device = torch.device('cpu')
traindataset_path = config.traindataset_path

with open(traindataset_path, "rb") as fp:
    traindataset = pickle.load(fp)
model_path = "./results/model.pth9"
model = NCFModel(config.hidden_dim, traindataset.user_nums,
                 traindataset.book_nums)
model.load_state_dict(torch.load(model_path))  # 导入网络的参数

df = pd.read_csv(config.test_data_path)
user_for_test = df['user_id'].tolist()

predict_item_id = []


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


BATCH_SIZE = 512
f = open(config.submit_data_path, 'w', encoding='utf-8')
f.write("user_id,item_id\n")
for user_id in user_for_test:
    #将用户已经交互过的物品排除
    user_visited_items = traindataset.user_book_map[user_id]
    items_for_predict = list(
        set(range(traindataset.book_nums)) - set(user_visited_items))
    results = []

    for batch in chunks(items_for_predict, BATCH_SIZE):
        user = torch.full([len(batch)], user_id).to(torch.int64).to(device)
        item = torch.Tensor(batch).to(dtype=torch.int64).to(device)
        result = model(user, item).view(-1).detach().cpu()
        results.append(result)

    results = torch.cat(results, dim=-1)
    predict_item_id = (-results).argsort()[:10]
    tep_string = ""
    for x in predict_item_id:
        tep_string += '{},{}\n'.format(user_id, x)
    f.write(tep_string)
    if user_id % 100 == 0:
        print(tep_string)
        f.flush()
f.flush()
f.close()