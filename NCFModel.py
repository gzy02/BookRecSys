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


class NCFModel(torch.nn.Module):

    def __init__(self, hidden_dim, user_num, item_num, mlp_layer_num, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.user_num = user_num
        self.item_num = item_num
        self.mlp_layer_num = mlp_layer_num
        self.dropout = dropout

        self.mlp_user_embedding = torch.nn.Embedding(
            user_num, hidden_dim * (2**(self.mlp_layer_num - 1)))
        self.mlp_item_embedding = torch.nn.Embedding(
            item_num, hidden_dim * (2**(self.mlp_layer_num - 1)))

        self.gmf_user_embedding = torch.nn.Embedding(user_num, hidden_dim)
        self.gmf_item_embedding = torch.nn.Embedding(item_num, hidden_dim)

        mlp_Layers = []
        input_size = int(hidden_dim * (2**(self.mlp_layer_num)))
        for i in range(self.mlp_layer_num):
            mlp_Layers.append(
                torch.nn.Linear(int(input_size), int(input_size / 2)))
            mlp_Layers.append(torch.nn.Dropout(self.dropout))
            mlp_Layers.append(torch.nn.ReLU())
            input_size /= 2
        self.mlp_layers = torch.nn.Sequential(*mlp_Layers)
        self.output_layer = torch.nn.Linear(2 * self.hidden_dim, 1)

    def forward(self, x):
        user = x[:, 0]
        item = x[:, 1]
        user_gmf_embedding = self.gmf_user_embedding(user)
        item_gmf_embedding = self.gmf_item_embedding(item)

        user_mlp_embedding = self.mlp_user_embedding(user)
        item_mlp_embedding = self.mlp_item_embedding(item)

        gmf_output = user_gmf_embedding * item_gmf_embedding

        mlp_input = torch.cat([user_mlp_embedding, item_mlp_embedding], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)

        output = torch.sigmoid(
            self.output_layer(torch.cat([gmf_output, mlp_output],
                                        dim=-1))).squeeze(-1)
        return output

    def my_predict(self, user, item):  #不对forward进行sigmoid 防止全变成1了丧失排序的意义
        user_gmf_embedding = self.gmf_user_embedding(user)
        item_gmf_embedding = self.gmf_item_embedding(item)

        user_mlp_embedding = self.mlp_user_embedding(user)
        item_mlp_embedding = self.mlp_item_embedding(item)

        gmf_output = user_gmf_embedding * item_gmf_embedding

        mlp_input = torch.cat([user_mlp_embedding, item_mlp_embedding], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)

        return self.output_layer(torch.cat([gmf_output, mlp_output], dim=-1))

    def predict(self, user, item):
        self.eval()
        with torch.no_grad():
            user_gmf_embedding = self.gmf_user_embedding(user)
            item_gmf_embedding = self.gmf_item_embedding(item)

            user_mlp_embedding = self.mlp_user_embedding(user)
            item_mlp_embedding = self.mlp_item_embedding(item)

            gmf_output = user_gmf_embedding.unsqueeze(1) * item_gmf_embedding

            user_mlp_embedding = user_mlp_embedding.unsqueeze(1).expand(
                -1, item_mlp_embedding.shape[1], -1)
            mlp_input = torch.cat([user_mlp_embedding, item_mlp_embedding],
                                  dim=-1)
            mlp_output = self.mlp_layers(mlp_input)

        output = torch.sigmoid(
            self.output_layer(torch.cat([gmf_output, mlp_output],
                                        dim=-1))).squeeze(-1)
        return output


import config

if __name__ == "__main__":
    net = NCFModel(config.hidden_dim, 53423, 10000, config.mlp_layer_num,
                   config.dropout)
    summary(net, input_size=(2, 1))
