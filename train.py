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

from plot import plot_stats


def _clearline():
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE+ERASE_LINE+CURSOR_UP_ONE)


def generate_images(model,onehot,img_size=[28,28],cuda=True):
    model.eval()
    y = np.array(list(range(10))*5)
    gen = torch.from_numpy(np.zeros([y.size, 1]+img_size, dtype='float32'))
    y = onehot(y)
    if cuda:
        y, gen = y.cuda(), gen.cuda()
    y, gen = Variable(y), Variable(gen)
    bar = ProgressBar()
    print('Generating images...')
    for r in bar(range(img_size[0])):
        for c in range(img_size[1]):
            out = model(gen,y)
            p = torch.exp(out)[:,:,r,c]
            sample = p.multinomial(1)
            gen[:,:,r,c] = sample.float()/(out.shape[1]-1)
    _clearline()
    _clearline()
    return (255*gen.data.cpu().numpy()).astype('uint8')


def tile_images(imgs):
    n = len(imgs)
    h = imgs[0].shape[1]
    w = imgs[0].shape[2]
    r = int(np.floor(np.sqrt(n)))
    while n%r!=0:
        r -= 1
    c = int(n/r)
    imgs = np.squeeze(np.array(imgs),axis=1)
    imgs = np.transpose(imgs,(1,2,0))
    imgs = np.reshape(imgs,[h,w,r,c])
    imgs = np.transpose(imgs,(2,3,0,1))
    imgs = np.concatenate(imgs,1)
    imgs = np.concatenate(imgs,1)
    return imgs


def plot_loss(train_loss,val_loss):
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


def fit(train_loader,val_loader,model,exp_path,label_preprocess,onehot,loss_fcn,
        optimizer='adam',learnrate=1e-4,cuda=True,patience=20,max_epochs=200,
        resume=False):

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
            y = onehot(y)
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
        _clearline()
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

        # Generate images and save gif
        new_frame = tile_images(generate_images(model,onehot))
        generated.append(new_frame)
        imageio.mimsave(os.path.join(exp_path, 'generated.gif'),
                        np.array(generated), format='gif', loop=0, fps=2)

        # Save gif again with loss plot
        plot_frame = plot_loss(stats['loss']['train'],stats['loss']['val'])
        if new_frame.ndim==2:
            new_frame = np.repeat(new_frame[:,:,np.newaxis],3,axis=2)
        nw = int(new_frame.shape[1]*plot_frame.shape[0]/new_frame.shape[0])
        new_frame = resize(new_frame,[plot_frame.shape[0],nw],
                           order=0, preserve_range=True, mode='constant')
        plots.append(np.concatenate((plot_frame.astype('uint8'),
                                     new_frame.astype('uint8')),
                                    axis=1))
        imageio.mimsave(os.path.join(exp_path, 'generated_plot.gif'),
                        np.array(plots), format='gif', loop=0, fps=2)

        # Save gif frames so it can resume training if interrupted
        np.save(os.path.join(exp_path,'generated.npy'),generated)
        np.save(os.path.join(exp_path,'generated_plots.npy'),plots)

        # Save stats and update plots
        with open(statsfile,'w') as sf:
            json.dump(stats,sf)
        plot_stats(stats,exp_path)

        # Early stopping
        torch.save(model,os.path.join(exp_path,'last_checkpoint'))
        if loss<best_val:
            best_val = loss
            stall = 0
            torch.save(model,os.path.join(exp_path,'best_checkpoint'))
        else:
            stall += 1
        if stall>=patience:
            break





