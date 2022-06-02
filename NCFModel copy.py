import torch
import torch.nn
from summary import summary


class NCFModel(torch.nn.Module):

    def __init__(self,
                 hidden_dim,
                 user_num,
                 item_num,
                 mlp_layer_num=4,
                 weight_decay=1e-5,
                 dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.user_num = user_num
        self.item_num = item_num
        self.mlp_layer_num = mlp_layer_num
        self.weight_decay = weight_decay
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

        # return -r_pos_neg + reg
        return output

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

    def my_predict(self, user, item):
        user_gmf_embedding = self.gmf_user_embedding(user)
        item_gmf_embedding = self.gmf_item_embedding(item)

        user_mlp_embedding = self.mlp_user_embedding(user)
        item_mlp_embedding = self.mlp_item_embedding(item)

        gmf_output = user_gmf_embedding * item_gmf_embedding

        mlp_input = torch.cat([user_mlp_embedding, item_mlp_embedding], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)

        return self.output_layer(torch.cat([gmf_output, mlp_output], dim=-1))


import config

if __name__ == "__main__":
    net = NCFModel(config.hidden_dim, 53423, 10000)
    summary(net, input_size=(2, 1))
