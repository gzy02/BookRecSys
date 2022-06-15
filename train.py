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
from Goodbooks import Goodbooks

traindataset_path = config.traindataset_path
validdataset_path = config.validdataset_path
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
    traindataset = Goodbooks(df, user_book_map, 'training')

    with open(traindataset_path, "wb") as fp:
        pickle.dump(traindataset, fp)
else:
    with open(traindataset_path, "rb") as fp:
        traindataset = pickle.load(fp)

#不存在就建立，否则直接load
if not os.path.exists(validdataset_path):
    validdataset = Goodbooks(df, user_book_map, 'validation')

    with open(validdataset_path, "wb") as fp:
        pickle.dump(validdataset, fp)
else:
    with open(validdataset_path, "rb") as fp:
        validdataset = pickle.load(fp)

trainloader = DataLoader(traindataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         drop_last=False,
                         num_workers=0)
validloader = DataLoader(validdataset,
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
from MFModel import MFModel
from NCFModel import NCFModel
from Goodbooks import get_adj_mat
from LightGCN import LightGCN
if config.use_ncf:
    model = NCFModel(hidden_dim, traindataset.user_nums,
                     traindataset.book_nums, mlp_layer_num, dropout).to(device)
    if config.is_load_model:  #如果导入已经训练了的模型
        model.load_state_dict(torch.load(config.load_model_path))
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay)
elif config.use_mf:
    model = MFModel(hidden_dim, traindataset.user_nums,
                    traindataset.book_nums).to(device)
    if config.is_load_model:  #如果导入已经训练了的模型
        model.load_state_dict(torch.load(config.load_model_path))
    #optimizer = torch.optim.Adam(model.parameters(),
    #                             lr=learning_rate,
    #                             weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay)
elif config.use_gcn:
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

crit = torch.nn.BCELoss()  #损失函数：BCELoss

loss_for_plot = []
hits_for_plot = []

for epoch in range(epochs):

    #在训练集上训练
    losses = []
    for data in trainloader:
        user, item, label = data
        x = torch.stack((user, item), dim=1)
        x, label = x.to(device), label.to(device).float()
        y_ = model(x).squeeze()

        loss = crit(y_, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().item())

    #在验证集上验证
    hits = []
    hits_n = 10
    for data in validloader:
        user, pos, neg = data
        pos = pos.unsqueeze(1)
        all_data = torch.cat([pos, neg], dim=-1)
        output = model.predict(user.to(device),
                               all_data.to(device)).detach().cpu()

        for batch in output:
            if 0 not in (-batch).argsort()[:hits_n]:
                hits.append(0)
            else:
                hits.append(1)

    print('Epoch {} finished, average loss {}, hits@{} {}'.format(
        epoch,
        sum(losses) / len(losses), hits_n,
        sum(hits) / len(hits)))
    loss_for_plot.append(sum(losses) / len(losses))
    hits_for_plot.append(sum(hits) / len(hits))

    # 模型保存
    model_path = config.model_path
    hits_for_plot_path = config.hits_for_plot_path
    loss_for_plot_path = config.loss_for_plot_path
    torch.save(model.state_dict(),
               model_path + str(epoch + config.load_model_epoch + 1))

    hits_for_plot_past = []
    loss_for_plot_past = []
    if config.is_load_model == True:
        with open(hits_for_plot_path, "rb") as fp:
            hits_for_plot_past = pickle.load(fp)
        with open(loss_for_plot_path, "rb") as fp:
            loss_for_plot_past = pickle.load(fp)
    #跟之前的历史数据拼起来，再存储
    hits_for_plot_past += hits_for_plot
    loss_for_plot_past += loss_for_plot
    with open(hits_for_plot_path, "wb") as fp:
        pickle.dump(hits_for_plot_past, fp)
    with open(loss_for_plot_path, "wb") as fp:
        pickle.dump(hits_for_plot_past, fp)
