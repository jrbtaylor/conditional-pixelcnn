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
5. Closing remarks: it'd be interesting to see if anyone has combined PixelCNN and GANs, using the PixelCNN as a prior or a final stage of the decoder (conditioned on some higher-level learned representation) to avoid some of the training difficulties with GANs; Should note that PixelCNNs have more potential than GANs for use outside of generating flashy images for social media, e.g. detecting out-of-domain images for a main network (especially if it has an attention model for checking if poorly-predicted pixels are important for the main model) or adversarial examples (though they likely suffer similar limitations)

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
![masked-convolution blind spot](http://github.com/jrbtaylor/conditional-pixelcnn/docs/blindspot.png "blind spot")

## PixelCNNs vs GANs

PixelCNNs and GANs are the two flavors of deep learning models for generating images.
Currently GANs are receiving a lot of attention, but in many ways I find their popularity unwarranted.
It's unclear what objective GANs are actually trying to optimize or if the minimum of this objective generalizes to examples outside of the training set.
This is reflected in the notorious difficulty of training GANs.
The idea of pitting two nets against each other to produce training signals is interesting and has produced many good papers (notably cycleGAN)
but I remain unconvinced that they'll be useful for much beyond making flashy posts on social media.
On the other hand, PixelCNNs have a probabilistic underpinning.
This allows them to not only generate images by sampling the distribution (left-to-right, top-to-bottom, following their autoregressive definition), but also means they can be used for other tasks as an auxiliary/pre-screening network to detect out-of-domain or adversarial examples, or to model uncertainty.
I'll cover these extensions more in a later post.

## Implementation

My implementation uses the gated blocks but for rapid implementation, I decided to forego the two-stream solution to the blind spot problem (separating the filters into horizontal and vertical components).