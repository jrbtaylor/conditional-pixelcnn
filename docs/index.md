---
layout: page
title: Conditional PixelCNN in PyTorch
tagline: 
description: A PyTorch implementation of Conditional PixelCNNs to generate between-class examples, tested on MNIST
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
___
