#5.模型预测
#加载测试数据
#生成提交文件

import pandas as pd
import pickle
import config
import torch
from NCFModel import NCFModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')
print(device)
traindataset_path = config.traindataset_path
with open(traindataset_path, "rb") as fp:
    traindataset = pickle.load(fp)
model = NCFModel(config.hidden_dim,
                 traindataset.user_nums,
                 traindataset.book_nums,
                 mlp_layer_num=config.mlp_layer_num,
                 dropout=0)  #dropout

df = pd.read_csv(config.test_data_path)
user_for_test = df['user_id'].tolist()

BATCH_SIZE = 512


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


import heapq


def main(it: int):

    model_path = f"./models/model.pth{it}"
    submission_path = f"./submit/submit{it}.csv"
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    f = open(submission_path, 'w', encoding='utf-8')
    f.write("user_id,item_id\n")
    for user_id in user_for_test:
        #将用户已经交互过的物品排除
        user_visited_items = traindataset.user_book_map[user_id]
        items_for_predict = list(
            set(range(traindataset.book_nums)) - set(user_visited_items))

        results = []

        for batch in chunks(items_for_predict, BATCH_SIZE):
            user = torch.full([len(batch)],
                              user_id).to(dtype=torch.int64).to(device)
            item = torch.Tensor(batch).to(dtype=torch.int64).to(device)
            result = model.my_predict(user, item).cpu()
            for i in range(len(batch)):
                results.append((batch[i], result[i]))

        predict_item_id = []
        largest_10 = heapq.nlargest(10, results, key=lambda x: x[1])
        for i in largest_10:
            predict_item_id.append(i[0])
        #results.sort(key=lambda x: x[1], reverse=True)
        #for i in range(10):
        #    predict_item_id.append(results[i][0])

        tep_string = ""
        for x in predict_item_id:
            tep_string += '{},{}\n'.format(user_id, x)
        f.write(tep_string)
        if user_id % 100 == 0:
            print(tep_string)
            f.flush()
    f.flush()
    f.close()


if __name__ == "__main__":
    main(26)