import torch
import torch.nn as nn
from summary import summary


class DNNModel(nn.Module):

    def __init__(self, user_num, item_num, hidden_dim, mlp_layer_num):
        super(DNNModel, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.mlp_layer_num = mlp_layer_num
        self.hidden_dim = hidden_dim
        self.user_embed = nn.Embedding(
            self.user_num, self.hidden_dim * (2**(self.mlp_layer_num - 1)))
        self.item_embed = nn.Embedding(
            self.item_num, self.hidden_dim * (2**(self.mlp_layer_num - 1)))

        self.user_dnn = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(),
                                      nn.Linear(64, 64), nn.LeakyReLU())

        self.item_dnn = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(),
                                      nn.Linear(64, 64), nn.LeakyReLU())

    def forward(self, x):
        u = self.user_embed(x[:, 0])
        m = self.item_embed(x[:, 1])
        u = self.user_dnn(u)
        m = self.item_dnn(m)
        u = u / torch.sum(u * u, 1).view(-1, 1)
        m = m / torch.sum(m * m, 1).view(-1, 1)
        return u, m


import config

if __name__ == "__main__":
    net = DNNModel(
        53423,
        10000,
        config.hidden_dim,
        mlp_layer_num=4,
    )
    summary(net, input_size=(2, 1))
