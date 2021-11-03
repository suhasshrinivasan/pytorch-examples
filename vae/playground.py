#%%
from __future__ import print_function
import argparse
import PIL
from PIL.Image import ANTIALIAS
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import _is_numpy_image
from torchvision.utils import save_image

#%%
seed = 42
torch.manual_seed(seed)
device = torch.device("cuda")
kwargs = {'num_workers': 1, 'pin_memory': True}
batch_size = 128
#%%
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=False,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

# %%
"""
data str:
dataloader = {
    'train' = {
        'sessionID': pytorch.dataloader
    }
    'val' = {

    }
    'test' = {

    }
}

dataloader['train'].values()
"""

a = {
    'a': [1, 2],
    'b': [3, 4, 5]
}
#%%
print(*a.values())


# %%
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
first_session_id = list(dataloaders['train'].keys())[0]
train_loader_first_session = dataloaders['train'][first_session_id]
train_loader = train_loader_first_session
# train dataset fixed.

# test dataset to be fixed now
# test data batching is done differently remember -- each batch in the test set is purely repeats.
# hence from each test batch, pick only one image tensor
# start with the first session
test_loader_first_session = dataloaders['test'][first_session_id]
testset_images = [inputs[0] for inputs, targets in test_loader_first_session]
test_loader = DataLoader(testset_images)
#%%
from PIL.Image import NEAREST, BILINEAR

resizer = transforms.Resize(size=(28, 28), interpolation=NEAREST)

#%%
torch.__version__
#%%
# construct a data (image) resizer
import matplotlib.pyplot as plt
# sample some resized images and take a look
for batch_idx, (data, _) in enumerate(train_loader):
    # plt.imshow(data[0,0,::])
    plt.figure(figsize=(10,10))
    plt.imshow(data[0].permute(1, 2, 0))
    plt.show()
    data_resized = resizer(data)
    plt.figure(figsize=(1,1))
    plt.imshow(data_resized[0].permute(1, 2, 0))
    plt.show()
    print(data_resized[0].shape)
    if batch_idx > 5:
        break


# %%
import numpy as np

# %%
def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample
    

dataset = datasets.DatasetFolder(
    root='../../../data/monkey/toliaslab/CSRF19_V1/images/',
    loader=npy_loader,
    extensions=('.npy')
)
# %%
import torch
from torch.nn import Linear

a = Linear(400, 20)
# %%
b = torch.randn(400)
c = a(b)
# %%
dir(c)
# %%
type(c)
# %%

# %%
