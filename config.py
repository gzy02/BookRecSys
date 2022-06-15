BATCH_SIZE = 512
hidden_dim = 24
epochs = 100
weight_decay = 0
dropout = 0
mlp_layer_num = 6
learning_rate = 0.001

use_ncf = False
use_mf = False
use_gcn = True

is_load_model = False
load_model_epoch = 400

if is_load_model == False:
    load_model_epoch = 0

validdataset_path = "./pkl/validdataset.pkl"
user_book_map_path = "./pkl/user_book_map.pkl"
submit_data_path = './submit/submission.csv'
train_data_path = './datasets/train_dataset.csv'
test_data_path = './datasets/test_dataset.csv'

hits_for_plot_path = "./pkl/hits_for_plot.pkl"
loss_for_plot_path = "./pkl/loss_for_plot.pkl"
traindataset_path = "./pkl/traindataset.pkl"
model_path = './models/model.pth'
load_model_path = f'./models/model.pth{load_model_epoch}'
