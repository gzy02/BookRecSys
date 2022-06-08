use_fake_data = True  #是否使用添加了fake label的数据

BATCH_SIZE = 512
hidden_dim = 128
epochs = 50
weight_decay = 0
dropout = 0.5
mlp_layer_num = 6
learning_rate = 5e-4

use_ncf = False
use_mf = True

is_load_model = False
load_model_epoch = 0

if is_load_model == False:
    load_model_epoch = 0

validdataset_path = "./pkl/validdataset.pkl"
user_book_map_path = "./pkl/user_book_map.pkl"
submit_data_path = './submit/submission.csv'
train_data_path = './datasets/train_dataset.csv'
test_data_path = './datasets/test_dataset.csv'

if use_fake_data == False:
    hits_for_plot_path = "./pkl/no_fake_hits_for_plot.pkl"
    loss_for_plot_path = "./pkl/no_fake_loss_for_plot.pkl"
    traindataset_path = "./pkl/traindataset_no_fake.pkl"
    model_path = './no_fake_model/model.pth'
    load_model_path = f'./no_fake_model/model.pth{load_model_epoch}'
else:
    hits_for_plot_path = "./pkl/hits_for_plot.pkl"
    loss_for_plot_path = "./pkl/loss_for_plot.pkl"
    traindataset_path = "./pkl/traindataset.pkl"
    model_path = './models/model.pth'
    load_model_path = f'./model/model.pth{load_model_epoch}'
