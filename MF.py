import torch
import torch.nn as nn


class MF(nn.Module):

    def __init__(self, embedding_size, num_users, num_items, mean=1):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)

        self.user_emb.weight.data.uniform_(0, 0.005)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(0, 0.005)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        # 全局bias
        self.mean = nn.Parameter(torch.FloatTensor([mean]), False)

    def forward(self, x):
        u_id = x[:, 0]
        i_id = x[:, 1]
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        return torch.sigmoid((U * I).sum(1) + b_u + b_i + self.mean)

    def my_predict(self, u_id, i_id):  #不对forward进行sigmoid 防止全变成1了丧失排序的意义
        self.eval()  #不dropout
        with torch.no_grad():  #不求导
            U = self.user_emb(u_id)
            b_u = self.user_bias(u_id).squeeze()
            I = self.item_emb(i_id)
            b_i = self.item_bias(i_id).squeeze()
        return (U * I).sum(1) + b_u + b_i + self.mean