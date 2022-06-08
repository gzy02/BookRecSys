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

        # ȫ��bias
        self.mean = nn.Parameter(torch.FloatTensor([mean]), False)

    def forward(self, x):
        u_id = x[:, 0]
        i_id = x[:, 1]
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        return torch.sigmoid((U * I).sum(1) + b_u + b_i + self.mean)

    def my_predict(self, u_id, i_id):  #����forward����sigmoid ��ֹȫ���1��ɥʧ���������
        self.eval()  #��dropout
        with torch.no_grad():  #����
            U = self.user_emb(u_id)
            b_u = self.user_bias(u_id).squeeze()
            I = self.item_emb(i_id)
            b_i = self.item_bias(i_id).squeeze()
        return (U * I).sum(1) + b_u + b_i + self.mean