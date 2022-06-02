# 图书推荐系统
# 任务描述：建立一个隐式推荐算法，可以预测用户交互的下一本书。
# 数据集:使用了Goodbooks-10k数据集，包含了10000本图书和53424个用户共约6M条交互记录。
# 方法概述：首先加载数据，并划分训练集和验证集。搭建NCF(Neural Collaborative Filtering)模型，并构建负样本，最终按照模型输出的评分进行排序，做出最终的推荐。
# baseline:https://work.datafountain.cn/forum?id=563&type=2&source=1
# %% [1]
import config
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import os
#%% [2]

BATCH_SIZE = config.BATCH_SIZE
hidden_dim = config.hidden_dim
epochs = config.epochs
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device(
    'cpu')
print(device)
# %%[3]
# 1.数据准备
#1.1 数据说明
#1.2 数据预处理

#
# 1.1 加载数据
#观察样本数量和稀疏度。
train_data_path = config.train_data_path
df = pd.read_csv(train_data_path)
print('共{}个用户，{}本图书，{}条记录'.format(
    max(df['user_id']) + 1,
    max(df['item_id']) + 1, len(df)))
df.head()

import tqdm
from Goodbooks import Goodbooks

# %% [6]
import pickle
#建立训练和验证dataloader

traindataset_path = config.traindataset_path
validdataset_path = config.validdataset_path
user_book_map_path = config.user_book_map_path

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

if not os.path.exists(traindataset_path):
    traindataset = Goodbooks(df, user_book_map, 'training')

    with open(traindataset_path, "wb") as fp:
        pickle.dump(traindataset, fp)
else:
    with open(traindataset_path, "rb") as fp:
        traindataset = pickle.load(fp)

if not os.path.exists(validdataset_path):
    validdataset = Goodbooks(df, user_book_map, 'validation')

    with open(validdataset_path, "wb") as fp:
        pickle.dump(validdataset, fp)
else:
    with open(validdataset_path, "rb") as fp:
        validdataset = pickle.load(fp)

print("load pkl success")

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

# %% [7]
#2.模型构建
#NCF模型由GMF部分和MLP部分组成。
#
#Embedding Layer: 嵌入层，将稀疏的one-hot用户/物品向量转化为稠密的低维向量
#GMF Layer: 通过传统的矩阵分解算法，将以用户和物品的嵌入向量做内积。有效地提取浅层特征。
#MLP Layer: 通过n层全连接层，提取深层特征。
#Concatenation Layer: 将GMF和MLP输出的结果做concat，结合其中的深层和浅层信息。
#Output Layer: 输出层，输出用户-物品对的最终评分。
# 构建模型
# %% [8]
#3.模型训练&4.模型评估
#训练策略
#训练模型，固定步数会计算准确率
#模型保存
#可视化训练过程，对比训练集和验证集的准确率
from NCFModel import NCFModel

model = NCFModel(hidden_dim, traindataset.user_nums,
                 traindataset.book_nums).to(device)
if config.is_load_model:
    model.load_state_dict(torch.load(config.load_model_path))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = torch.nn.BCELoss()

loss_for_plot = []
hits_for_plot = []

for epoch in range(epochs):
    losses = []
    for index, data in enumerate(trainloader):
        user, item, label = data
        user, item, label = user.to(device), item.to(device), label.to(
            device).float()
        y_ = model(user, item).squeeze()

        loss = crit(y_, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().item())

    hits = []
    for index, data in enumerate(validloader):
        user, pos, neg = data
        pos = pos.unsqueeze(1)
        all_data = torch.cat([pos, neg], dim=-1)
        output = model.predict(user.to(device),
                               all_data.to(device)).detach().cpu()

        for batch in output:
            if 0 not in (-batch).argsort()[:10]:
                hits.append(0)
            else:
                hits.append(1)
    print('Epoch {} finished, average loss {}, hits@10 {}'.format(
        epoch,
        sum(losses) / len(losses),
        sum(hits) / len(hits)))
    loss_for_plot.append(sum(losses) / len(losses))
    hits_for_plot.append(sum(hits) / len(hits))

    # 模型保存
    model_path = config.model_path
    hits_for_plot_path = config.hits_for_plot_path
    loss_for_plot_path = config.loss_for_plot_path
    torch.save(model.state_dict(),
               model_path + str(epoch + config.load_model_epoch + 1))
    with open(hits_for_plot_path, "wb") as fp:
        pickle.dump(hits_for_plot, fp)
    with open(loss_for_plot_path, "wb") as fp:
        pickle.dump(loss_for_plot, fp)
