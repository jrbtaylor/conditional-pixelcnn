# Conditional PixelCNN

## TO-DO:
1. Move label formatting to data.py (in dataset target transform)
2. Add additional datasets (http://pytorch.org/docs/master/torchvision/datasets.html#fashion-mnist)
    * Fashion MNIST
    * EMNIST (letters)
3. Try training again without up-weighting 1's (might be over-weighting the grays in between)
4. Add data augmentation (http://pytorch.org/docs/master/torchvision/transforms.html)
    * add to data.py (in dataset transform)
    * random resize (h and w independent) then random crop/pad
5. Finish writing blog post
6. Write a real README file