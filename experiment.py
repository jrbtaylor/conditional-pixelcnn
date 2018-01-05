"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2018-2019
"""

import argparse
import json
import os

import torch
import numpy as np

import data
import model
import train

def run(dataset='mnist', batch_size=64, n_features=150, n_layers=6, n_bins=4,
        optimizer='adam', learnrate=1e-4, dropout=0.5, exp_name='pixelCNN',
        exp_dir='/home/jason/experiments/conditional-pixelcnn/',
        n_classes=10, cuda=True, resume=False):

    exp_name += '_%s_%ifeat_%ilayers_%ibins'%(
        dataset, n_features, n_layers, n_bins)
    exp_dir = os.path.join(exp_dir, exp_name)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)

    if not resume:
        # Store experiment params in params.json
        params = {'batch_size':batch_size, 'n_features':n_features,
                  'n_layers':n_layers, 'n_bins':n_bins, 'optimizer': optimizer,
                  'learnrate':learnrate, 'dropout':dropout, 'cuda':cuda}
        with open(os.path.join(exp_dir,'params.json'),'w') as f:
            json.dump(params,f)

        # Model
        net = model.PixelCNN(1, n_classes, n_features, n_layers, n_bins,
                             dropout)
    else:
        # if resuming, need to have params, stats and checkpoint files
        if not (os.path.isfile(os.path.join(exp_dir,'params.json'))
                and os.path.isfile(os.path.join(exp_dir,'stats.json'))
                and os.path.isfile(os.path.join(exp_dir,'last_checkpoint'))):
            raise Exception('Missing param, stats or checkpoint file on resume')
        net = torch.load(os.path.join(exp_dir, 'last_checkpoint'))

    # Data loaders
    train_loader,val_loader = data.loader(dataset, batch_size)

    # Up-weight 1s (~8x rarer) to balance loss, interpolate intermediate values
    weight = torch.from_numpy(np.linspace(1, 8, n_bins, dtype='float32'))
    if cuda:
        weight = weight.cuda()

    # Define loss fcn, incl. label formatting from input
    def input2label(x):
        return torch.squeeze(torch.round((n_bins-1)*x).type(torch.LongTensor),1)
    loss_fcn = torch.nn.NLLLoss2d(torch.autograd.Variable(weight))

    # Train
    train.fit(train_loader, val_loader, net, exp_dir, input2label, loss_fcn,
              n_classes, optimizer, learnrate=learnrate, cuda=cuda,
              resume=resume)