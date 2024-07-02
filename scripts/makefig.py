import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse

parser = argparse.ArgumentParser(description='calc L2 norms and generate figures')
parser.add_argument('--run_name', type=str, default='')
parser.add_argument('--path', type=str, default='/scratch/network/ls1546/mine-pytorch/scripts/')
parser.add_argument('--dir', type=str, default='/scratch/network/ls1546/mine-pytorch/scripts/')
parser.add_argument('--image_dim', type=int, default=224//5)
parser.add_argument('--num_neurons', type=int, default=4)

args = parser.parse_args()

run_name = args.run_name
path = args.path
dir = args.dir
image_dim = args.image_dim
num_neurons = args.num_neurons

# load ground truth weights
flat_weights = np.load(f'{path}{run_name}_flat_weights.npy')
num_true_filters = len(flat_weights)

def scale_rgb(x):
    return (x - x.min()) / (x.max() - x.min())

# send in flattened weights for a neuron, split into 3 rgb channels, and scale rgb vals
def get_c(flat_x, image_dim):
    x = np.reshape(flat_x, (3, image_dim, image_dim))
    for c in range(3):
        x[c] = scale_rgb(x[c])

    return x

# takes in the filter weights from T network and the ground truth weights, returns the L2 norm btw the pixels
# send in normalized filter weights for one neuron
def L2norm_filter(filter, gr_truth):
    return np.sum((gr_truth - filter)**2)

Tweights = []
l2norms = []

# calculate the l2norms between the T filters and the ground truth filters
# l2norms shape: [results for each run of T (based on num neurons), filter num in T, filter num in ground truth]
for i in range(1, num_neurons+1):
    Tweights.append(np.load(f'{path}{run_name}_{i}_Tweights.npy'))
    T_neurons = []
    for n in range(len(Tweights[i-1])):
        temp = []
        for j in range(num_true_filters):
            temp.append(L2norm_filter(get_c(Tweights[i-1][n], image_dim),
                                       np.reshape(flat_weights[j], (3, image_dim, image_dim))))
        T_neurons.append(temp)
    l2norms.append(T_neurons)


# loads the loss, loss type, and then plots the loss, loss type, and T filters with the L2 norms for all T weights
for i in range(len(Tweights)):
    num_filters = len(Tweights[i])
    plt.figure()
    if i == 0:
        fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(15, 10))
    else:
        fig, ax = plt.subplots(2, num_filters, figsize=(15, 10))
    name = f"{run_name}, neurons: {i+1}"
    plt.suptitle(f"{name}")

    loss_list = np.load(f"{path}{run_name}_{i+1}_loss.npy")
    loss_type = np.load(f"{path}{run_name}_{i+1}_loss_type.npy")
    
    ax[0, 0].plot(loss_list)
    ax[0, 0].set_title("loss")
    ax[0, 0].set_ylabel("loss")
    ax[0, 0].set_xlabel("batches")

    ax[0, 1].plot(loss_type)
    ax[0, 1].set_title("loss type")
    ax[0, 1].set_ylabel("loss type (0=mine_biased, 1=mine)")
    ax[0, 1].set_xlabel("batches")

    for n in range(num_filters):
        ax[1, n].pcolormesh(
        np.transpose(get_c(Tweights[i][n], image_dim), (1, 2, 0)),
        edgecolors="k",
        linewidth=0.005,
        )
        ax[1, n].set_title(f"T net filter {n}")
        plt.gca()
        ax[1, n].set_aspect("equal")
        ax[1, n].annotate(f'L2 norm with ground truth 1: {l2norms[i][n][0]:.2f}',
                           (0,0), (0, -20), xycoords='axes fraction',
                             textcoords='offset points', va='top')
        ax[1, n].annotate(f'L2 norm with ground truth 2: {l2norms[i][n][1]:.2f}',
                           (0,0), (0, -30), xycoords='axes fraction',
                             textcoords='offset points', va='top')

        plt.savefig(f"{run_name}{n}_wgts_loss.pdf")