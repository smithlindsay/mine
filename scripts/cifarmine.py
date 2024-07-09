import torch.nn as nn
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
from tqdm.autonotebook import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import mine
import argparse

parser = argparse.ArgumentParser(description="from CNN neuron acts, plot MI vs num neurons in T net")

parser.add_argument('--path', type=str, default='/scratch/gpfs/ls1546/mine/', help='path to mine dir')
parser.add_argument('--batch_size', type=int, default=500, help='batch size')
parser.add_argument('--image_size', type=int, default=32, help='image size')
parser.add_argument('--num_neurons', type=int, default=10, help='num neurons in T net: arange(1, num_neurons)')
parser.add_argument('--epochs', type=int, default=350, help='num of epochs')
parser.add_argument('--run_name', type=str, default='newcifar', help='run name of acts')
args = parser.parse_args()

# params
path = args.path
datadir = f'{path}data/'
figdir = f'{path}figures/'
batch_size = args.batch_size
image_size = args.image_size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_neurons = args.num_neurons
neurons = np.arange(1, num_neurons)
epochs = args.epochs
x_dim=image_size*image_size*3
y_dim=1
run_name = args.run_name

class ImageNetwork(nn.Module):
    def __init__(self, x_dim, y_dim, i):
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
    
# split the responses and images into a train, val, and test set for a specific neuron (n_num)
def data_split(images_flat, act, n_num):
    train_samples = int(.8*len(images_flat))
    val_samples = int(.1*len(images_flat))

    images_flat_train = images_flat[:train_samples]
    act_train = act[:train_samples, n_num].reshape(-1, 1)

    images_flat_val = images_flat[train_samples:train_samples+val_samples]
    act_val = act[train_samples:train_samples+val_samples, n_num].reshape(-1, 1)

    images_flat_test = images_flat[train_samples+val_samples:]
    act_test = act[train_samples+val_samples:, n_num].reshape(-1, 1)

    img_train = torch.tensor(images_flat_train, dtype=torch.float32)
    act_train = torch.tensor(act_train, dtype=torch.float32)
    img_val = torch.tensor(images_flat_val, dtype=torch.float32)
    act_val = torch.tensor(act_val, dtype=torch.float32)
    img_test = torch.tensor(images_flat_test, dtype=torch.float32)
    act_test = torch.tensor(act_test, dtype=torch.float32)
    return img_train, act_train, img_val, act_val, img_test, act_test   
 
# load data
images_flat = np.load(f'{datadir}{run_name}_images.npy')
act_l1 = np.load(f'{datadir}{run_name}_act_l1.npy')
act_l2 = np.load(f'{datadir}{run_name}_act_l2.npy')
act_l3 = np.load(f'{datadir}{run_name}_act_l3.npy')

def train_mine(run, neurons, lam, img_train, act_train, epochs, batch_size, img_val, act_val):
    for i in neurons:
        model = mine.Mine(
            T=ImageNetwork(x_dim, y_dim, i),
            loss="mine",  # mine_biased, fdiv
            device=device).to(device)

        run_name = f"{run}_{neurons[i-1]}_{lam}"

        mi, loss_list, loss_type = model.optimize(img_train, act_train, epochs, batch_size, 
                                                lam, run_name, img_val, act_val)

        torch.save(model.T, f"{datadir}{run_name}_mine.pt")
        np.save(f"{datadir}{run_name}_mi.npy", mi.detach().cpu().numpy())
        np.save(f"{datadir}{run_name}_loss.npy", loss_list)
        np.save(f"{datadir}{run_name}_loss_type.npy", loss_type)

        plt.figure()
        plt.plot(loss_list)
        plt.title(f"loss: {run_name}, {epochs} epochs")
        plt.ylabel("loss")
        plt.xlabel("batches")
        plt.savefig(f"{figdir}{run_name}_loss.pdf")
        plt.close()

        plt.figure()
        plt.plot(loss_type)
        plt.title(f"loss type: {run_name}, {epochs} epochs")
        plt.ylabel("loss type (0=mine_biased, 1=mine)")
        plt.xlabel("batches")
        plt.savefig(f"{figdir}{run_name}_loss_type.pdf")
        plt.close()

        Tweights = model.T.fc1x.weight.detach().cpu().numpy()
        np.save(f'{datadir}{run_name}_Tweights.npy', Tweights)

# use train_mine to train mine on some neurons from each linear layer of the CNN
final_mi = np.empty((9, 3, 4)) # 9 mine neurons, 3 layers, 4 cnn neurons
lams = [0.005, 0.5, 0.05]
pos = [4, 5, 6, 7]
acts = [act_l1, act_l2, act_l3]

for l in range(1, 4):
    for n in pos:
        lam = lams[l-1]
        run = f'cifarminel{l}_n{n}'
        img_train, act_train, img_val, act_val, _, _ = data_split(images_flat, acts[l-1], n)
        train_mine(run, neurons, lam, img_train, act_train, epochs, batch_size, img_val, act_val)

colors = ['red', 'green', 'blue']

for i in range(1, 10):
    for l in range(1, 4):
        for p in range(4):
            final_mi[i-1, l-1, p] = np.load(f'{datadir}cifarminel{l}_n{pos[p]}_{i}_{lams[l-1]}_mi.npy')[0]


plt.close('all')
plt.figure()
for l in range(1, 4):
    for p in range(4):
        plt.plot(neurons, final_mi[:, l-1, p], label=f'l{l} n{pos[p]}', color=colors[l-1])

plt.ylabel('final MI')
plt.xlabel('number of neurons')
plt.title(f'final MI - linear layers 1-3 of CNN (neurons {*pos,})')
# put legend outside of plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(f'{figdir}final_mi_l1_l2_l3_v2.pdf')