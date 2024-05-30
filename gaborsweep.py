from mine.models.mine import Mine
import torch.nn as nn
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from math import ceil
import sys
from skimage.filters import gabor_kernel

# params
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
neurons = np.arange(1, 5)
# for AdamW
lam = 0.005
# crop images to image_dim x image_dim
image_dim = 224//5
run = "gabor1d"
epochs = 250

x_dim=((224//5))*((224//5))*3
y_dim=1

batch_size = 500

def scale_rgb(x):
    return (x - x.min()) / (x.max() - x.min())

images_flat = np.load('images_flat_gabor.npy')
responses1 = np.load('responses1_gabor.npy')
responses2 = np.load('responses2_gabor.npy')
outputs = np.load('outputs_gabor.npy')

# split the responses and images into a train and test set
train_samples = int(.9*len(images_flat))
images_flat_train = images_flat[:train_samples]
outputs_train = outputs[:train_samples]
images_flat_test = images_flat[train_samples:]
outputs_test = outputs[train_samples:]

for i in neurons:
    class Image_network(nn.Module):
        def __init__(self, x_dim, y_dim):
            super().__init__()
            self.fc1x = nn.Linear(x_dim, i, bias=False)
            self.fc1y = nn.Linear(y_dim, 1, bias=True)
            self.fc2 = nn.Linear((i+1), 100, bias=True)
            self.fc3 = nn.Linear(100, 1, bias=True)

        def forward(self, x, y):
            x = F.relu(self.fc1x(x))
            y = F.relu(self.fc1y(y))
            h = torch.cat((x, y), dim=1)
            h = F.relu(self.fc2(h))
            h = self.fc3(h)
            return h

    mine = Mine(
        T=Image_network(x_dim, y_dim),
        loss="mine",  # mine_biased, fdiv
        device=device).to(device)

    run_name = f"{run}_{neurons[i-1]}"
    mi, loss_list, loss_type = mine.optimize(torch.tensor(images_flat_train, dtype=torch.float32), torch.tensor(outputs_train, dtype=torch.float32), epochs, batch_size, lam, run_name, torch.tensor(images_flat_test, dtype=torch.float32), torch.tensor(outputs_test, dtype=torch.float32))

    torch.save(mine.T, f"{run_name}_mine.pth")
    np.save(f"{run_name}_mi.npy", mi.detach().cpu().numpy())
    np.save(f"{run_name}_loss.npy", loss_list)
    np.save(f"{run_name}_loss_type.npy", loss_type)

    plt.figure()
    plt.plot(loss_list)
    plt.title(f"loss: {run_name}, {epochs} epochs")
    plt.ylabel("loss")
    plt.xlabel("batches")
    plt.savefig(f"{run_name}_loss.pdf")

    plt.figure()
    plt.figure()
    plt.plot(loss_type)
    plt.title(f"loss type: {run_name}, {epochs} epochs")
    plt.ylabel("loss type (0=mine_biased, 1=mine)")
    plt.xlabel("batches")
    plt.savefig(f"{run_name}_loss_type.pdf")

    Tweights = mine.T.fc1x.weight.detach().cpu().numpy()
    np.save(f'{run_name}_Tweights.npy', Tweights)

    # unflat_Tweights = np.reshape(Tweights, (3, (224//5), (224//5)))

    # for i in range(3):
    #     plt.clf()
    #     plt.figure()
    #     plt.pcolormesh(scale_rgb(unflat_Tweights[i]), edgecolors="k", linewidth=0.005)
    #     ax = plt.gca()
    #     ax.set_aspect("equal")
    #     plt.colorbar()
    #     plt.title(f"{run_name}, channel {i}")
    #     plt.savefig(f"{run_name}_Tweightsc{i}.pdf")

    # plt.figure()
    # plt.pcolormesh(
    #     np.transpose(np.array(list(map(scale_rgb, unflat_Tweights))), (1, 2, 0)),
    #     edgecolors="k",
    #     linewidth=0.005,
    # )
    # plt.title(f"{run_name}, combined channels")
    # ax = plt.gca()
    # ax.set_aspect("equal")
    # plt.savefig(f"{run_name}_Tweightscomb.pdf")
    # plt.close()
