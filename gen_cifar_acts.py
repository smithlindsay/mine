import torch.nn as nn
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange
import argparse
import torch.optim as optim

parser = argparse.ArgumentParser(description='generate neuron activations from simple cifar10 classifier')
parser.add_argument('--run_name', type=str, default='')
parser.add_argument('--path', type=str, default='/scratch/gpfs/ls1546/mine/')
parser.add_argument('--dir', type=str, default='/scratch/gpfs/ls1546/mine/')
parser.add_argument('--image_dim', type=int, default=224//5)
parser.add_argument('--num_neurons', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=1000)

args = parser.parse_args()

# params
run_name = args.run_name
path = args.path
dir = args.dir
image_dim = args.image_dim
num_neurons = args.num_neurons
batch_size = args.batch_size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def scale_rgb(x):
    return (x - x.min()) / (x.max() - x.min())

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))])

# TODO: add val split to train data

traindata = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size,
                                          shuffle=True, num_workers=8, pin_memory=True)

testdata = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size,
                                         shuffle=False, num_workers=8, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

# create simple CNN to later get activations from (when trained)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 channels, 6 output chs, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2) # 2x2 pooling
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 in chs, 16 out chs, 5x5 kernel
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dim except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = CNN().to(device)

# train model
criterion = nn.CrossEntropyLoss()
optim = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

epochs = 10
for epoch in (pbar := tqdm(range(1, epochs + 1))):
    for images, labels in tqdm(trainloader):
        optim.zero_grad()
        outputs = net(images.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optim.step()
        pbar.set_description(f"epoch: {epoch}, loss: {loss.item():4f}")




flat_weights = []
flat_weights.append(flat_weights1)
flat_weights.append(flat_weights2)
np.save(f'{run_name}_flat_weights', flat_weights)


# add hooks, run model with inputs to get activations

# a dict to store the activations
activation = {}
def get_activation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach().cpu().numpy()
        # activation[name] = output.numpy()
    return hook

hook1 = net.fc1.register_forward_hook(get_activation('fc1'))
# hook2 = net.fc2.register_forward_hook(get_activation('fc2'))

inputs_list = []
# outputs_list = []
act_list1 = []
# act_list2 = []

# pass images through CNN to get activations
for inputs, _ in tqdm(dataloader):
    inputs = torch.flatten(inputs, start_dim=1)
    inputs = inputs.to(device)

    with torch.no_grad():
        output = net(inputs)
        
        # collect the activations
        act_list1.append(activation['fc1'])
        # act_list2.append(activation['fc2'])

        inputs_list.append(inputs.detach().cpu().numpy())
        # outputs_list.append(output.detach().cpu().numpy())

    del inputs
    del output

# detach the hooks
hook1.remove()
# hook2.remove()

act_length = (len(act_list1) - 1)*batch_size + len(act_list1[len(act_list1)-1])
samples = (len(inputs_list) - 1)*batch_size + len(inputs_list[len(inputs_list)-1])
images_flat = np.zeros((samples, ((224//5))*((224//5))*3))
responses1 = np.zeros((act_length, 1))
# responses2 = np.zeros((act_length, 1))
# outputs = np.zeros((act_length, 1))
x_dim=((224//5))*((224//5))*3
y_dim=1

for batch in range(len(act_list1)):
    for image in range(len(act_list1[batch])):
        responses1[batch*len(act_list1[0])+image, 0] = act_list1[batch][image, 0]
        # responses2[batch*len(act_list2[0])+image, 0] = act_list2[batch][image, 0]
        # outputs[batch*len(act_list1[0])+image, 0] = outputs_list[batch][image, 0]
        images_flat[batch*len(act_list1[0])+image, :] = inputs_list[batch][image]

# del act_list1, act_list2, inputs_list, outputs_list
del act_list1, inputs_list
# del act_list, inputs_list

np.save(f'images_flat_gabor_true', images_flat)
np.save(f'responses1_gabor_true', responses1)
# np.save(f'responses2_gabor', responses2)
# np.save(f'outputs_gabor', outputs)