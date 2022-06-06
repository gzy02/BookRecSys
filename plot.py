import matplotlib.pyplot as plt
import pickle
import config


def show():
    hits_for_plot_path = config.hits_for_plot_path
    loss_for_plot_path = config.loss_for_plot_path

    with open(hits_for_plot_path, "rb") as fp:
        hits_for_plot = pickle.load(fp)
    with open(loss_for_plot_path, "rb") as fp:
        loss_for_plot = pickle.load(fp)

    print("hits:", hits_for_plot)
    print("loss:", loss_for_plot)

    x = list(range(1, len(hits_for_plot) + 1))
    plt.subplot(1, 3, 1)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(x, loss_for_plot, 'r')

    plt.subplot(1, 3, 3)
    plt.xlabel('epochs')
    plt.ylabel('hits@10')
    plt.plot(x, hits_for_plot, 'r')

    plt.show()


if __name__ == "__main__":
    show()