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
from einops import rearrange
import argparse

parser = argparse.ArgumentParser(description='generate neuron activations from ground truth gabor filters')
parser.add_argument('--run_name', type=str, default='')
parser.add_argument('--path', type=str, default='/scratch/network/ls1546/mine-pytorch/scripts/')
parser.add_argument('--dir', type=str, default='/scratch/network/ls1546/mine-pytorch/scripts/')
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

# crop images to image_dim x image_dim
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(image_dim),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# USE 100K TEST DATA
dataset = torchvision.datasets.ImageFolder(
    root='/scratch/network/ls1546/imagenet/ILSVRC/Data/CLS-LOC/test', 
    transform=transform
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# make the gabor filter
wavelen = 224//10
frequency = 1/wavelen # signal completes 1 full cycle in wavelen pixels
theta = 45.0 # tilt of the gabor filter
# sigma_x and sigma_y constrain the std dev of the gaussian
kernel1 = gabor_kernel(frequency, theta=theta, sigma_x=13, sigma_y=13)
# up and down gabor
# kernel2 = gabor_kernel(frequency, theta=0.0, sigma_x=13, sigma_y=13)

input_height1, input_width1 = kernel1.shape
# input_height2, input_width2 = kernel2.shape

# Calculate cropping boundaries
crop_top1 = (input_height1 - image_dim) // 2
crop_bottom1 = crop_top1 + image_dim
crop_left1 = (input_width1 - image_dim) // 2
crop_right1 = crop_left1 + image_dim

# crop_top2 = (input_height2 - image_dim) // 2
# crop_bottom2 = crop_top2 + image_dim
# crop_left2 = (input_width2 - image_dim) // 2
# crop_right2 = crop_left2 + image_dim

# Perform the crop
weights1 = np.real(kernel1)[crop_top1:crop_bottom1, crop_left1:crop_right1]
# weights2 = np.real(kernel2)[crop_top2:crop_bottom2, crop_left2:crop_right2]

# scale weights
weights1 = scale_rgb(weights1)
# weights2 = scale_rgb(weights2)

# repeat the weights for each channel and flatten
weights1 = torch.tensor(weights1, dtype=torch.float32)
flat_weights1 = np.repeat(weights1.unsqueeze(0), 3, axis=0).view(1, -1)
# weights2 = torch.tensor(weights2, dtype=torch.float32)
# flat_weights2 = np.repeat(weights2.unsqueeze(0), 3, axis=0).view(1, -1)

# plt.figure()
# plt.imshow(weights1)
# ax = plt.gca()
# ax.set_aspect('equal')
# plt.colorbar()
# plt.title('ground truth gabor filter 1')
# plt.savefig(f"example_gabor1.pdf")

# plt.figure()
# plt.imshow(weights2)
# ax = plt.gca()
# ax.set_aspect('equal')
# plt.colorbar()
# plt.title('ground truth gabor filter 2')
# plt.savefig(f"example_gabor2.pdf")

flat_weights = []
flat_weights.append(flat_weights1)
flat_weights.append(flat_weights2)
np.save(f'{run_name}_flat_weights', flat_weights)

# pass images through toy_network to get activations
class Toynetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(image_dim*image_dim*3, 1)
        self.fc1.weight = torch.nn.Parameter(flat_weights1)
        # self.fc2 = nn.Linear(image_dim*image_dim*3, 1)
        # self.fc2.weight = torch.nn.Parameter(flat_weights2)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        # y = F.relu(self.fc2(input))
        return x

toy_net = Toynetwork().to(device)

# add hooks, run model with inputs to get activations

# a dict to store the activations
activation = {}
def get_activation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach().cpu().numpy()
        # activation[name] = output.numpy()
    return hook

hook1 = toy_net.fc1.register_forward_hook(get_activation('fc1'))
# hook2 = toy_net.fc2.register_forward_hook(get_activation('fc2'))

inputs_list = []
# outputs_list = []
act_list1 = []
# act_list2 = []

for inputs, _ in tqdm(dataloader):
    inputs = torch.flatten(inputs, start_dim=1)
    inputs = inputs.to(device)

    with torch.no_grad():
        output = toy_net(inputs)
        
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