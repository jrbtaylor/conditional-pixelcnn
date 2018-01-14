"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2018-2019
"""

import imageio
import json
import os
from PIL import Image
import time

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from progressbar import ProgressBar
from skimage.transform import resize
import torch
from torch.autograd import Variable

from vis import plot_stats, clearline, generate, tile_images


def generate_images(model,img_size,n_classes,onehot_fcn,cuda=True):
    y = np.array(list(range(n_classes))*5)
    y = np.concatenate([onehot_fcn(x)[np.newaxis,:] for x in y])
    return generate(model, img_size, y, cuda)


def plot_loss(train_loss, val_loss):
    fig = plt.figure(num=1, figsize=(4, 4), dpi=70, facecolor='w',
                     edgecolor='k')
    plt.plot(range(1,len(train_loss)+1), train_loss, 'r', label='training')
    plt.plot(range(1,len(val_loss)+1), val_loss, 'b', label='validation')
    plt.title('After %i epochs'%len(train_loss))
    plt.xlabel('Epoch')
    plt.ylabel('Cross-entropy loss')
    plt.rcParams.update({'font.size':10})
    fig.tight_layout(pad=1)
    fig.canvas.draw()

    # now convert the plot to a numpy array
    plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    plot = plot.reshape(fig.canvas.get_width_height()[::-1]+(3,))
    plt.close(fig)
    return plot


def fit(train_loader, val_loader, model, exp_path, label_preprocess, loss_fcn,
        onehot_fcn, n_classes=10, optimizer='adam', learnrate=1e-4, cuda=True,
        patience=10, max_epochs=200, resume=False):

    if cuda:
        model = model.cuda()

    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)
    statsfile = os.path.join(exp_path,'stats.json')

    optimizer = {'adam':torch.optim.Adam(model.parameters(),lr=learnrate),
                 'sgd':torch.optim.SGD(
                     model.parameters(),lr=learnrate,momentum=0.9),
                 'adamax':torch.optim.Adamax(model.parameters(),lr=learnrate)
                 }[optimizer.lower()]

    # load a single example from the iterator to get the image size
    x = train_loader.sampler.data_source.__getitem__(0)[0]
    img_size = list(x.numpy().shape[1:])

    if not resume:
        stats = {'loss':{'train':[],'val':[]},
                 'mean_output':{'train':[],'val':[]}}
        best_val = np.inf
        stall = 0
        start_epoch = 0
        generated = []
        plots = []
    else:
        with open(statsfile,'r') as js:
            stats = json.load(js)
        best_val = np.min(stats['loss']['val'])
        stall = len(stats['loss']['val'])-np.argmin(stats['loss']['val'])-1
        start_epoch = len(stats['loss']['val'])-1
        generated = list(np.load(os.path.join(exp_path,'generated.npy')))
        plots = list(np.load(os.path.join(exp_path,'generated_plot.npy')))
        print('Resuming from epoch %i'%start_epoch)

    def save_img(x,filename):
        Image.fromarray((255*x).astype('uint8')).save(filename)

    def epoch(dataloader,training):
        bar = ProgressBar()
        losses = []
        mean_outs = []
        for x,y in bar(dataloader):
            label = label_preprocess(x)
            if cuda:
                x,y = x.cuda(),y.cuda()
                label = label.cuda()
            x,y = Variable(x),Variable(y)
            label = Variable(label)
            if training:
                optimizer.zero_grad()
                model.train()
            else:
                model.eval()
            output = model(x,y)
            loss = loss_fcn(output,label)
            # track mean output
            output = output.data.cpu().numpy()
            mean_outs.append(np.mean(np.argmax(output,axis=1))/output.shape[1])
            if training:
                loss.backward()
                optimizer.step()
            losses.append(loss.data.cpu().numpy())
        clearline()
        return float(np.mean(losses)), np.mean(mean_outs)

    for e in range(start_epoch,max_epochs):
        # Training
        t0 = time.time()
        loss,mean_out = epoch(train_loader,training=True)
        time_per_example = (time.time()-t0)/len(train_loader.dataset)
        stats['loss']['train'].append(loss)
        stats['mean_output']['train'].append(mean_out)
        print(('Epoch %3i:    Training loss = %6.4f    mean output = %1.2f    '
               '%4.2f msec/example')%(e,loss,mean_out,time_per_example*1000))

        # Validation
        t0 = time.time()
        loss,mean_out = epoch(val_loader,training=False)
        time_per_example = (time.time()-t0)/len(val_loader.dataset)
        stats['loss']['val'].append(loss)
        stats['mean_output']['val'].append(mean_out)
        print(('            Validation loss = %6.4f    mean output = %1.2f    '
               '%4.2f msec/example')%(loss,mean_out,time_per_example*1000))

        # Generate images and update gif
        new_frame = tile_images(generate_images(model, img_size, n_classes,
                                                onehot_fcn, cuda))
        generated.append(new_frame)

        # Update gif with loss plot
        plot_frame = plot_loss(stats['loss']['train'],stats['loss']['val'])
        if new_frame.ndim==2:
            new_frame = np.repeat(new_frame[:,:,np.newaxis],3,axis=2)
        nw = int(new_frame.shape[1]*plot_frame.shape[0]/new_frame.shape[0])
        new_frame = resize(new_frame,[plot_frame.shape[0],nw],
                           order=0, preserve_range=True, mode='constant')
        plots.append(np.concatenate((plot_frame.astype('uint8'),
                                     new_frame.astype('uint8')),
                                    axis=1))

        # Save gif arrays so it can resume training if interrupted
        np.save(os.path.join(exp_path,'generated.npy'),generated)
        np.save(os.path.join(exp_path,'generated_plots.npy'),plots)

        # Save stats and update training curves
        with open(statsfile,'w') as sf:
            json.dump(stats,sf)
        plot_stats(stats,exp_path)

        # Early stopping
        torch.save(model,os.path.join(exp_path,'last_checkpoint'))
        if loss<best_val:
            best_val = loss
            stall = 0
            torch.save(model,os.path.join(exp_path,'best_checkpoint'))
            imageio.imsave(os.path.join(exp_path, 'best_generated.jpeg'),
                           generated[-1].astype('uint8'))
            imageio.imsave(os.path.join(exp_path, 'best_generated_plots.jpeg'),
                           plots[-1].astype('uint8'))
            imageio.mimsave(os.path.join(exp_path, 'generated.gif'),
                            np.array(generated), format='gif', loop=0, fps=2)
            imageio.mimsave(os.path.join(exp_path, 'generated_plot.gif'),
                            np.array(plots), format='gif', loop=0, fps=2)
        else:
            stall += 1
        if stall>=patience:
            break






