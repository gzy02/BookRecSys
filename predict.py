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

model = NCFModel(config.hidden_dim, traindataset.user_nums,
                 traindataset.book_nums)
model.load_state_dict(torch.load(config.model_path))  # 导入网络的参数

df = pd.read_csv(config.test_data_path)
user_for_test = df['user_id'].tolist()

predict_item_id = []


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


f = open(config.submit_data_path, 'w', encoding='utf-8')
for user in user_for_test:
    #将用户已经交互过的物品排除
    user_visited_items = traindataset.user_book_map[user]
    items_for_predict = list(
        set(range(traindataset.book_nums)) - set(user_visited_items))
    #items_for_predict = np.array(items_for_predict).reshape(1, -1)
    results = []
    user = torch.Tensor([user]).to(torch.int64).cpu()
    for batch in items_for_predict:  #chunks(items_for_predict, 64):
        batch = torch.Tensor([batch]).to(torch.int64).cpu()
        result = model(user, batch).view(-1).detach().cpu()
        results.append(result)

    results = torch.cat(results, dim=-1)
    predict_item_id = (-results).argsort()[:10]
    for x in predict_item_id:
        f.write('{},{}\n'.format(user.cpu().item(), x))

f.close()