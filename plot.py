hits_for_plot_path = "./hits_for_plot.pkl"
loss_for_plot_path = "./loss_for_plot.pkl"
import pickle
with open(hits_for_plot_path, "rb") as fp:
    hits_for_plot = pickle.load(fp)
with open(loss_for_plot_path, "rb") as fp:
    loss_for_plot = pickle.load(fp)

import matplotlib.pyplot as plt

x = list(range(1, len(hits_for_plot) + 1))
plt.subplot(1, 2, 1)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(x, loss_for_plot, 'r')

plt.subplot(1, 2, 2)
plt.xlabel('epochs')
plt.ylabel('acc')
plt.plot(x, hits_for_plot, 'r')

plt.show()