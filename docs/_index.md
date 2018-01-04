---
title: Conditional PixelCNNs
tagline: 
description: A PyTorch implementation of Conditional PixelCNNs to generate between-class examples
---

outline:
1. Motivation: want to learn PyTorch; want to see if (believable) between-class examples can be generated with a conditional PixelCNN (maybe to train a better classifier)
2. PyTorch basics (i.e. dynamic computer graphs, no sessions/compiling)
3. Conditional PixelCNN: PixelCNN intro; PixelCNN vs GAN (Bayesian vs ill-defined objective); didn't fix blindspot problem (for quick implementation) but added gated activations and conditional vector; code snippets as needed to explain/highlight things; hyperparams
4. Results: training curve and final output (0-9); between-class examples for 1/7, 3/8, 4/9, 5/6 on a continuum of soft-labels (1-0,0.9-0.1,0.8-0.2,...,0-1)
5. Discussion: mention possible use of generated between-class examples for a learned mixup (rather than averaging pixels), though this would require multiple GPUs working to generate batches of new examples for training the classifier

## markdown cheatsheet:
*italics*
**bold**
~~strikethrough~~
[inline-style link](https://www.google.com)
![image name](https://imageurl "mouse-over text")
```python
s = "Python syntax highlighting"
print s
```

horizontal rule is 3 or more underscores


## Motivation
This is the first of what I expect will be many posts on machine learning.
I started a [machine learning blog on WordPress](http://netsprawl.wordpress.com) in 2017 but abandoned it 2 posts in after finding that showing code without messing up the formating was not possible - the forced narrow column format would wrap the code and render it unreadable.

For 2018, my new year's resolution is to write 5 posts (as Github project pages). I wanted to play with PixelCNNs and finally try [PyTorch](http://pytorch.org) (I use Tensorflow for my work at [Envision.AI](http://envision.ai) and previously used Theano at McGill) so this post will include my thoughts on both.
In particular, this post will focus on generating between-class examples.




## Conditional PixelCNNs

PixelCNNs are the convolutional version of PixelRNNs, which treat the pixels in an image as a time-series and predict each pixel after seeing the preceding pixels (above and to the left).
PixelRNNs are an autoregressive model of the joint prior distribution for images:

<p style="text-align: center;"> p(x) = p(x<sub>0</sub>) &prod; p(x<sub>i</sub>| x<sub>i<</sub>) </p>

PixelRNNs are slow to train since the recurrence can't be parallelized. 
Replacing the model recurrence with masked convolutions, where the convolution filter only sees pixels above and to the left, allows for faster training.
However, it's worth noting that the [original PixelCNN implementation](https://arxiv.org/abs/1601.06759) produced worse results than the PixelRNN.
One possible reason for the degraded results, conjectured in the [follow-up paper](https://arxiv.org/abs/1606.05328), is the relative simplicity of the ReLU activations in the PixelCNN compared to the gated connections in the LSTM.
The Conditional PixelCNN paper subsequently replaced the ReLUs with gated activations:
<p style="text-align: center;"> y = <i>tanh</i>(W<sub>f</sub>&lowast; x) &bull; &sigmaf;(W<sub>g</sub>&lowast; x) </p>
Another possible reason offered in the follow-up paper is that stacking masked convolutional filters results in blind spots, failing to capture all the pixels above the one being predicted:
![masked-convolution blind spot](https://github.com/jrbtaylor/conditional-pixelcnn/blob/master/docs/blindspot.png?raw=true)




#### PixelCNNs vs GANs

PixelCNNs and GANs are the two (current) flavors of deep learning models for generating images.
Recently GANs are receiving a lot of attention, but in many ways I find their popularity unwarranted.

It's unclear what objective GANs are actually trying to optimize and the minimum of the training objective (fooling the discriminator) would result in the generator recreating all the training images.
This is reflected in the notorious difficulty of training GANs.
The idea of pitting two nets against each other to produce training signals is interesting and has produced many good papers (notably cycleGAN)
but I remain unconvinced that they'll be useful for much beyond making flashy posts on social media.

On the other hand, PixelCNNs have a probabilistic underpinning.
This allows them to not only generate images by sampling the distribution (left-to-right, top-to-bottom, following their autoregressive definition), 
but also means they can be used for other tasks as an auxiliary/pre-screening network to detect out-of-domain or adversarial examples (especially if the main network has attention: the PixelCNN error can be weighted per-pixel by importantance to the main model).
Going one step further, it may also be possible to estimate uncertainty for new examples with a PixelCNN trained on the same distribution as the main network.
I'll cover some of these extensions more in a later post.

I'd be interested in hearing if anyone has tried combining PixelCNNs and GANs. Perhaps the PixelCNN can be used as a prior or a final stage of the decoder (conditioned on some higher-level learned representation) to avoid some of the training difficulties with GANs.




## Implementation

My implementation uses the gated blocks but for rapid implementation, I decided to forego the two-stream solution to the blind spot problem (separating the filters into horizontal and vertical components).
This way the masking is simple: everything below and to the right of the current pixel is zeroed-out in the filter and in the first layer the current pixel is also set to zero in the filter.
```python
class MaskedConv(nn.Conv2d):
    def __init__(self,mask_type,in_channels,out_channels,kernel_size,stride=1):
        super(MaskedConv,self).__init__(in_channels,out_channels,kernel_size,
                                        stride,padding=kernel_size//2)
        assert mask_type in ('A','B')
        mask = torch.ones(1,1,kernel_size,kernel_size)
        mask[:,:,kernel_size//2,kernel_size//2+(mask_type=='B'):] = 0
        mask[:,:,kernel_size//2+1:] = 0
        self.register_buffer('mask',mask)

    def forward(self,x):
        self.weight.data *= self.mask
        return super(MaskedConv,self).forward(x)
```
The implementation for the gated ResNet blocks is slightly more complicated:
the PixelCNN has shortcut connections between the two halfs of the network, like a U-Net;
PyTorch allows the forward method of a Module to take multiple inputs ONLY if they're Variables;
since the feature maps from the first half of the network are not Variables, they must be concatenated with the other input (the features from the preceding layer).
This messiness is avoided with the conditioning vector, since it is a Variable (in this case, the class label).
```python
class GatedRes(nn.Module):
    def __init__(self,in_channels,out_channels,n_classes,kernel_size=3,stride=1,
                 aux_channels=0):
        super(GatedRes,self).__init__()
        self.conv = MaskedConv('A',in_channels,2*out_channels,kernel_size,
                               stride)
        self.y_embed = nn.Linear(n_classes,2*out_channels)
        self.out_channels = out_channels
        if aux_channels!=2*out_channels and aux_channels!=0:
            self.aux_shortcut = nn.Sequential(
                nn.Conv2d(aux_channels,2*out_channels,1),
                nn.BatchNorm2d(2*out_channels,momentum=0.1))
        if in_channels!=out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,1),
                nn.BatchNorm2d(out_channels,momentum=0.1))
        self.batchnorm = nn.BatchNorm2d(out_channels,momentum=0.1)

    def forward(self,x,y):
        # check for aux input from first half of net stacked into x
        if x.dim()==5:
            x,aux = torch.split(x,1,dim=0)
            x = torch.squeeze(x,0)
            aux = torch.squeeze(x,0)
        else:
            aux = None
        x1 = self.conv(x)
        y = torch.unsqueeze(torch.unsqueeze(self.y_embed(y),-1),-1)
        if aux is not None:
            if hasattr(self,'aux_shortcut'):
                aux = self.aux_shortcut(aux)
            x1 = (x1+aux)/2
        # split for gate (note: pytorch dims are [n,c,h,w])
        xf,xg = torch.split(x1,self.out_channels,dim=1)
        yf,yg = torch.split(y,self.out_channels,dim=1)
        f = torch.tanh(xf+yf)
        g = torch.sigmoid(xg+yg)
        if hasattr(self,'shortcut'):
            x = self.shortcut(x)
        return x+self.batchnorm(g*f)
```
I wasn't sure where to put batch normalization from reading the original papers, so I placed it where I thought it made sense: prior to adding the residual connection.

With those two classes implemented, the full network was relatively easy. 
The PyTorch scheme of defining everything as subclasses of `nn.Module`, initializing all the layers/operations/etc. in the constructor and then connecting them together in the `forward` method can be messy.
This is especially true if you have lots of shortcut connections and want to code your model with loops for arbitrary depth. 

*Note:* to be able to save/restore the model, you have to store layers in a `ModuleList` instead of a regular list.
 Appending and indexing this list is otherwise the same though.

```python
class PixelCNN(nn.Module):
    def __init__(self,in_channels,n_classes,n_features,n_layers,n_bins,
                 dropout=0.5):
        super(PixelCNN,self).__init__()

        self.layers = nn.ModuleList()
        self.n_layers = n_layers

        # Up pass
        self.input_batchnorm = nn.BatchNorm2d(in_channels,momentum=0.1)
        for l in range(n_layers):
            if l==0:  # start with normal conv
                block = nn.Sequential(
                    MaskedConv('A',in_channels+1,n_features,kernel_size=7),
                    nn.BatchNorm2d(n_features,momentum=0.1),
                    nn.ReLU())
            else:
                block = GatedRes(n_features, n_features, n_classes)
            self.layers.append(block)

        # Down pass
        for _ in range(n_layers):
            block = GatedRes(n_features, n_features,n_classes,
                             aux_channels=n_features)
            self.layers.append(block)

        # Last layer: project to n_bins (output is [-1, n_bins, h, w])
        self.layers.append(
            nn.Sequential(nn.Dropout2d(dropout),
                          nn.Conv2d(n_features,n_bins,1),
                          nn.LogSoftmax(dim=1)))

    def forward(self,x,y):
        # Add channel of ones so network can tell where padding is
        x = nn.functional.pad(x,(0,0,0,0,0,1,0,0),mode='constant',value=1)

        # Up pass
        features = []
        i = -1
        for _ in range(self.n_layers):
            i += 1
            if i>0:
                x = self.layers[i](x,y)
            else:
                x = self.layers[i](x)
            features.append(x)

        # Down pass
        for _ in range(self.n_layers):
            i += 1
            x = self.layers[i](torch.stack((x,features.pop())),y)

        # Last layer
        i += 1
        x = self.layers[i](x)
        assert i==len(self.layers)-1
        assert len(features)==0
        return x
```
MNIST is practically black and white, so I discretized the label to only 8 grayscale levels for the purposes of calculating cross-entropy loss.
On natural images, the number of output levels would obviously need to be higher.
All layers in my network have 200 features.
For training, I used Adam with a learning rate of 10<sup>-4</sup> and dropout rate of 0.5.

I have a single GTX1070 GPU at home, so I didn't run any kind of hyperparameter optimization:
the ability to guess reasonable hyperparameters and have your model work on the first attempt says a lot about the robustness of Adam + batch normalization + dropout.




## Results





## Miscellaneous Thoughts on PyTorch
1. Debugging something without a compiled computational graph (i.e. like Tensorflow or Theano) is a lot faster
2. Writing models where you have to initialize every operation/layer/etc. in the constructor and then call them in the forward method seems unnecessarily complicated. This is especially problematic for models with shortcut connections if you write them with loops for arbitrary depth. This cancels out item 1.
3. 
