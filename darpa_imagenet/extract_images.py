from galatea.darpa_imagenet.utils import count_images
from galatea.darpa_imagenet.utils import ImageIterator
from galatea.darpa_imagenet.dataset import DARPA_ImageNet
import numpy as np
from pylearn2.utils import serial
from pylearn2.datasets.preprocessing import ExtractPatches

path = '/data/lisatmp/glorotxa/train'
image_shape = (32,32)
patch_shape = (6,6)
iterator = ImageIterator(path)

m = count_images(path)

batch_size = 50000
k = 3

Xs = []

rng = np.random.RandomState([1,2,3])

for i in xrange(0,m,batch_size):
    sub_m = min(batch_size, m-i)

    d = DARPA_ImageNet(iterator = iterator, num_examples = sub_m, image_shape = image_shape)

    random_rng = np.random.RanndomState([ rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)])

    p = ExtractPatches( patch_shape = patch_shape, num_patches = k * batch_size, rng = random_rng)

    d.apply_preprocessor(p)

    Xs.append(d.X)

X = np.concatenate(Xs,axis=0)

d.X = X

base = '/data/listamp/goodfeli/darpa/imagenet/img_%dx%d_patch_%dx%d_train.' % (image_shape[0], image_shape[1], patch_shape[0], patch_shape[1])

d.use_design_loc(base+'npy')
serial.save(base+'pkl')


