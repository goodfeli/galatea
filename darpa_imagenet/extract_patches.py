from galatea.darpa_imagenet.utils import count_images
from galatea.darpa_imagenet.utils import ImageIterator
import numpy as np
from pylearn2.utils import serial
from pylearn2.datasets.preprocessing import ExtractPatches
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter

path = '/data/lisatmp/goodfeli/darpa_imagenet'
image_shape = (32,32)
patch_shape = (6,6)

m = count_images(path,'.npy')
k = 3

X = np.zeros((m*k,patch_shape[0]*patch_shape[1]*3),dtype='float32')

rng = np.random.RandomState([1,2,3])

for i, img_path in enumerate(ImageIterator(path, suffix=".npy")):

    img = np.load(img_path)

    if img.shape[2] == 1:
        img = np.concatenate((img,img,img),axis=2)

    img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])

    d = DenseDesignMatrix( topo_view = img, view_converter = DefaultViewConverter(img.shape[1:]) )

    random_rng = np.random.RandomState([ rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)])

    p = ExtractPatches( patch_shape = patch_shape, num_patches = k , rng = random_rng)

    d.apply_preprocessor(p)

    X[i*3:(i+1)*3,:] = d.X

d.X = X

base = '/data/lisatmp/goodfeli/darpa_imagenet_patch_%dx%d_train.' % (patch_shape[0], patch_shape[1])

d.use_design_loc(base+'npy')
serial.save(base+'pkl',d)


