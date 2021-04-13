import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_range(array, label=None):
    mean = array.mean(axis=-1)
    std = array.std(axis=-1)

    plt.plot(mean, label=label)
    plt.fill_between(np.arange(len(array)), mean - 3 * std, mean + 3 * std, alpha=0.3)


def compute_cost(model, dataloader, loss_function=None):
    loss_function = loss_function or torch.nn.functional.mse_loss
    running_loss = 0
    with torch.no_grad():
        for u, y in dataloader:
            u = u.transpose(0, 1)
            y = y.transpose(0, 1)

            netout = model(u)

            running_loss += loss_function(netout.squeeze(), y.squeeze())
    return running_loss / len(dataloader.dataset)


def freq_filter(arr, alpha=0.99):
    def gen_array(arr, alpha):
        current = arr[0]
        for elt in arr:
            current = alpha * current + (1 - alpha) * elt
            yield current

    return np.array(list(gen_array(arr, alpha)))
