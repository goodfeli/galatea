from pylearn2.utils.serial import load

dataset = load('/u/goodfeli/galatea/datasets/norb_tiny_preprocessed_train.pkl')

X = dataset.get_topological_view(dataset.X)

import numpy as np
X /= np.abs(X).mean()
X += 1.
X /= 2.

img = X.mean(axis=0)

from pylearn2.utils.image import show

show(img)
