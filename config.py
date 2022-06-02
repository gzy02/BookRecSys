hits_for_plot_path = "./pkl/hits_for_plot.pkl"
loss_for_plot_path = "./pkl/loss_for_plot.pkl"
traindataset_path = "./pkl/traindataset.pkl"
validdataset_path = "./pkl/validdataset.pkl"
user_book_map_path = "./pkl/user_book_map.pkl"

submit_data_path = './submit/submission.csv'
train_data_path = './datasets/train_dataset.csv'
test_data_path = './datasets/test_dataset.csv'
model_path = './models/model.pth'

BATCH_SIZE = 512
hidden_dim = 16
epochs = 40
weight_decay = 1e-5
mlp_layer_num = 4
dropout = 0.5
learning_rate = 1e-3

is_load_model = False
load_model_epoch = 0
load_model_path = f'./results/model.pth{load_model_epoch}'

if is_load_model == False:
    load_model_epoch = 0
