import torch
import torch.nn
from summary import summary
#模型构建
#NCF模型由GMF部分和MLP部分组成。
#
#Embedding Layer: 嵌入层，将稀疏的one-hot用户/物品向量转化为稠密的低维向量
#GMF Layer: 通过传统的矩阵分解算法，将以用户和物品的嵌入向量做内积。有效地提取浅层特征。
#MLP Layer: 通过n层全连接层，提取深层特征。
#Concatenation Layer: 将GMF和MLP输出的结果做concat，结合其中的深层和浅层信息。
#Output Layer: 输出层，输出用户-物品对的最终评分。


class MFModel(torch.nn.Module):

    def __init__(self, hidden_dim, user_num, item_num):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.user_num = user_num
        self.item_num = item_num
        self.gmf_user_embedding = torch.nn.Embedding(user_num, hidden_dim)
        self.gmf_item_embedding = torch.nn.Embedding(item_num, hidden_dim)

    def forward(self, x):
        user = x[:, 0]
        item = x[:, 1]
        user_gmf_embedding = self.gmf_user_embedding(user)
        item_gmf_embedding = self.gmf_item_embedding(item)

        gmf_output = torch.mm(user_gmf_embedding,
                              item_gmf_embedding.T)  #对角线的就是有用的东西
        diag = torch.diag(gmf_output)
        output = torch.sigmoid(diag).squeeze(-1)
        return output

    def my_predict(self, user, item):  #不对forward进行sigmoid 防止全变成1了丧失排序的意义
        self.eval()  #不dropout
        with torch.no_grad():  #不求导
            user_gmf_embedding = self.gmf_user_embedding(user)
            item_gmf_embedding = self.gmf_item_embedding(item)

            gmf_output = torch.mm(user_gmf_embedding,
                                  item_gmf_embedding.T)  #对角线的就是有用的东西
            diag = torch.diag(gmf_output)
        return diag

    def predict(self, user, items):
        self.eval()
        outputs = torch.zeros(items.shape[0],
                              items.shape[1])  #(batch_size,100)
        with torch.no_grad():  #不求导
            for i in range(items.shape[1]):
                item = items[:, i]
                user_gmf_embedding = self.gmf_user_embedding(user)
                item_gmf_embedding = self.gmf_item_embedding(item)

                gmf_output = torch.mm(user_gmf_embedding,
                                      item_gmf_embedding.T)  #对角线的就是有用的东西
                diag = torch.diag(gmf_output)
                output = torch.sigmoid(diag).squeeze(-1)
                outputs[:, i] = output.T

        return outputs


import config

if __name__ == "__main__":
    net = MFModel(config.hidden_dim, 53423, 10000)
    summary(net, input_size=(2, ))
