from json import load

hits_for_plot_path = "./hits_for_plot.pkl"
loss_for_plot_path = "./loss_for_plot.pkl"
traindataset_path = "./traindataset.pkl"
validdataset_path = "./validdataset.pkl"
user_book_map_path = "./user_book_map.pkl"

submit_data_path = './results/submission.csv'
train_data_path = './datasets/train_dataset.csv'
test_data_path = './datasets/test_dataset.csv'
model_path = './results/model.pth'

BATCH_SIZE = 512
hidden_dim = 16
epochs = 40

is_load_model = True
load_model_epoch = 38
load_model_path = f'./results/model.pth{load_model_epoch}'
