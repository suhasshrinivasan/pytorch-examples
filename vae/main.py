#%%
from __future__ import print_function
import argparse
from re import L
from skimage import transform
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import _is_numpy_image
from torchvision.utils import save_image
#%%
# only for CLI running
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#%%
# for jupyter style running
class Args:
    def __init__(self):
        self.batch_size = 128
        self.cuda = True
        self.log_interval = 10
        self.epochs = 10
        self.seed = 12345

args = Args()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


#%%
# default example of MNIST
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


#%%
# neural data

import nnfabrik
from nnfabrik import builder


import numpy as np
import pickle
import os

from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt

import nnvision

basepath = '/home/data/monkey/toliaslab/CSRF19_V1'
neuronal_data_path = os.path.join(basepath, 'neuronal_data/')
neuronal_data_files = [neuronal_data_path+f for f in listdir(neuronal_data_path) if isfile(join(neuronal_data_path, f))]
image_file = os.path.join(basepath, 'images/CSRF19_V1_images.pickle')
image_cache_path = os.path.join(basepath, 'images/individual')

dataset_fn = 'nnvision.datasets.monkey_static_loader'
dataset_config = dict(dataset='CSRF19_V1',
                               neuronal_data_files=neuronal_data_files,
                               image_cache_path=image_cache_path,
                               crop=0,
                               subsample=1,
                               seed=1000,
                               time_bins_sum=6,
                               batch_size=128,)

dataloaders = builder.get_data(dataset_fn, dataset_config)

some_image = dataloaders["train"][list(dataloaders["train"].keys())[11]].dataset[:].inputs[0,0,::].cpu().numpy()
plt.imshow(some_image, cmap='gray')

#%%
# pick the first session
first_session_id = list(dataloaders['train'].keys())[0]
train_loader_first_session = dataloaders['train'][first_session_id]
train_loader = train_loader_first_session
# train dataset fixed.

#%%
# test dataset to be fixed now
# test data batching is done differently remember -- each batch in the test set is purely repeats.
# hence from each test batch, pick only one image tensor
# start with the first session
test_loader_first_session = dataloaders['test'][first_session_id]
testset_images = [inputs[0] for inputs, targets in test_loader_first_session]
test_loader = DataLoader(testset_images)
#%%
# construct a data (image) resizer
resizer = transforms.Resize(size=(28, 28))

#%%
import matplotlib.pyplot as plt
# sample some resized images and take a look
for batch_idx, (data, _) in enumerate(train_loader):
    # plt.imshow(data[0,0,::])
    plt.imshow(data[0].permute(1, 2, 0))
    plt.show()
    data_resized = resizer(data)
    # plt.figure(figsize=(2,2))
    plt.imshow(data_resized[0].permute(1, 2, 0))
    plt.show()
    print(data_resized[0].shape)
    if batch_idx > 5:
        break


#%%
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784)) # x.view(-1, 784) flattens x
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


#%%

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def sampling_loss_function(recon_x, x, mu, logvar, r):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # log posterior distribution function evaluated for neuronal responses
    # assuming q to be gaussian with diagonal covariance
    log_response_posterior = -0.5 * torch.det(logvar.exp()) - 0.5 * np.log(2*np.pi) - 0.5 * (r - mu).T @ logvar.exp().inverse() @ (r - mu)

    return log_response_posterior + BCE + KLD


    
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (stimulus, response) in enumerate(train_loader):
        stimulus = resizer(stimulus)
        stimulus = stimulus.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(stimulus)
        loss = sampling_loss_function(recon_batch, stimulus, mu, logvar, response)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(stimulus), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(stimulus)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, stimulus, response in enumerate(test_loader):
            stimulus = resizer(stimulus)
            stimulus = stimulus.to(device)
            recon_batch, mu, logvar = model(stimulus)

            test_loss += loss_function(recon_batch, stimulus, mu, logvar).item()
            if i == 0:
                n = min(stimulus.size(0), 8)
                comparison = torch.cat([stimulus[:n],
                                      recon_batch.view(1, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results_neural_imgs/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

#%%
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                    'results_neural_imgs/sample_' + str(epoch) + '.png')



#%%

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
