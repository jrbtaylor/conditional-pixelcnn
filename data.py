"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2018-2019
"""

import os

import numpy as np
import torch
from torchvision import datasets, transforms


def onehot(n_classes):
    def onehot_fcn(x):
        y = np.zeros((n_classes), dtype='float32')
        y[x] = 1
        return y
    return onehot_fcn


def loader(dataset,batch_size,n_workers=4):
    assert dataset.lower() in ['mnist','emnist','fashionmnist']

    loader_args = {'batch_size':batch_size, 'num_workers':n_workers,
                   'pin_memory':True}
    datapath = os.path.join(os.getenv('HOME'),'data',dataset.lower())
    dataset_args = {'root':datapath, 'download':True,
                    'transform':transforms.ToTensor()}

    if dataset.lower()=='mnist':
        dataset_init = datasets.MNIST
        n_classes = 10
    elif dataset.lower()=='emnist':
        dataset_init = datasets.EMNIST
        n_classes = 62
        dataset_args.update({'split':'byclass'})
    else:
        dataset_init = datasets.FashionMNIST
        n_classes = 10
    dataset_args.update({'target_transform':onehot(n_classes)})

    train_loader = torch.utils.data.DataLoader(
        dataset_init(train=True, **dataset_args), shuffle=True, **loader_args)
    val_loader = torch.utils.data.DataLoader(
        dataset_init(train=False, **dataset_args), shuffle=False, **loader_args)

    return train_loader, val_loader