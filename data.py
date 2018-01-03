"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2018-2019
"""

import torch
from torchvision import datasets, transforms

def mnist(batch_size,n_workers=4):
    kwargs = {'num_workers': n_workers, 'pin_memory': True}
    transform = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/home/jason/data/mnist',
                       train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/home/jason/data/mnist',
                       train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader,val_loader