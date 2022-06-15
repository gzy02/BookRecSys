# 图书推荐系统
# 任务描述：建立一个隐式推荐算法，可以预测用户交互的下一本书。
# 数据集:使用了Goodbooks-10k数据集，包含了10000本图书和53424个用户共约6M条交互记录。
# 方法概述：首先加载数据，并划分训练集和验证集。搭建NCF(Neural Collaborative Filtering)模型，并构建负样本，最终按照模型输出的评分进行排序，做出最终的推荐。
# baseline:https://work.datafountain.cn/forum?id=563&type=2&source=1
# %% [1]
import config
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader
import os

BATCH_SIZE = config.BATCH_SIZE
hidden_dim = config.hidden_dim
epochs = config.epochs
mlp_layer_num = config.mlp_layer_num
dropout = config.dropout
weight_decay = config.weight_decay
learning_rate = config.learning_rate

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')
print(device)

train_data_path = config.train_data_path
df = pd.read_csv(train_data_path)
print('共{}个用户，{}本图书，{}条记录'.format(
    max(df['user_id']) + 1,
    max(df['item_id']) + 1, len(df)))

# %%
# 数据预处理
# 构建Dataset类
# 构建负样本
# 划分测试集与验证集
# 构建对应的Dataloader
#
#建立训练和验证dataloader
from Goodbooks import Goodbooks_GCN

traindataset_path = "./pkl/GCN_TrainSet.pkl"
user_book_map_path = config.user_book_map_path
#不存在就建立，否则直接load
if not os.path.exists(user_book_map_path):
    user_book_map = {}
    for i in range(max(df['user_id']) + 1):
        user_book_map[i] = []

    for index, row in df.iterrows():
        user_id, book_id = row
        user_book_map[user_id].append(book_id)

    with open(user_book_map_path, "wb") as fp:
        pickle.dump(user_book_map, fp)
else:
    with open(user_book_map_path, "rb") as fp:
        user_book_map = pickle.load(fp)

#不存在就建立，否则直接load
if not os.path.exists(traindataset_path):
    traindataset = Goodbooks_GCN(df, user_book_map)

    with open(traindataset_path, "wb") as fp:
        pickle.dump(traindataset, fp)
else:
    with open(traindataset_path, "rb") as fp:
        traindataset = pickle.load(fp)

trainloader = DataLoader(traindataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         drop_last=False,
                         num_workers=0)

print("Load dataset success")

# %%
#模型训练&评估
#训练策略
#训练模型，固定步数会计算准确率
#模型保存
#可视化训练过程，对比训练集和验证集的准确率

from Goodbooks import get_adj_mat
from time import time

drop_flag = False
loss_type = 'bpr'
from LightGCN import LightGCN
if config.use_gcn:
    plain_adj, norm_adj, mean_adj = get_adj_mat(traindataset.user_nums,
                                                traindataset.book_nums,
                                                user_book_map)
    model = LightGCN(traindataset.user_nums, traindataset.book_nums,
                     norm_adj).to(device)
    if config.is_load_model:  #如果导入已经训练了的模型
        model.load_state_dict(torch.load(config.load_model_path))
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(),
    #                            lr=learning_rate,
    #                            weight_decay=weight_decay)
else:
    model = None


def train():
    loss = []
    mf_loss = []
    emb_loss = []
    for epoch in range(epochs):
        t1 = time()
        for data in trainloader:
            users, pos_items, neg_items = data
            batch_loss, batch_mf_loss, batch_emb_loss = getattr(
                model, f"create_{loss_type}_loss")(users.numpy().tolist(),
                                                   pos_items.numpy().tolist(),
                                                   neg_items.numpy().tolist(),
                                                   drop_flag=drop_flag)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss.append(batch_loss)
            mf_loss.append(batch_mf_loss)
            emb_loss.append(batch_emb_loss)

        perf_str = 'Epoch %d [%.1fs]: loss==[%.5f=%.5f + %.5f]' % \
                (epoch+1, time() - t1, sum(loss)/len(loss), sum(mf_loss)/len(mf_loss), sum(emb_loss)/len(emb_loss))
        print(perf_str)

        # 模型保存
        model_path = config.model_path
        loss_for_plot_path = config.loss_for_plot_path
        torch.save(model.state_dict(),
                   model_path + str(epoch + config.load_model_epoch + 1))

        loss_for_plot_past = []
        if config.is_load_model == True:
            with open(loss_for_plot_path, "rb") as fp:
                loss_for_plot_past = pickle.load(fp)
        #跟之前的历史数据拼起来，再存储
        loss_for_plot_past += loss
        with open(loss_for_plot_path, "wb") as fp:
            pickle.dump(loss_for_plot_past, fp)


if __name__ == "__main__":
    train()