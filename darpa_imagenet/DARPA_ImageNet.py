from galatea.darpa_imagenet.utils import count_images
from galatea.darpa_imagenet.utils import explore_images
from galatea.darpa_imagenet.utils import make_letterboxed_thumbnail
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.utils import image


import numpy as np

class DARPA_ImageNet_Dataset(DenseDesignMatrix):
    def __init__(self, path, image_shape):
        assert len(image_shape) == 2
        print 'Counting images...'
        m = count_images(path)
        print 'There are ',m,' images'

        T = np.zeros((m,image_shape[0],image_shape[1],3),dtype='float32')

        for i, img_path in explore_images(path):
            print 'Loading image ',m
            img = image.load(path)
            T[i,:] = make_letterboxed_thumbnail(img, image_shape)

        super(DARPA_ImageNet_Dataset, self).__init__(
                topo_view = T,
                view_converter = DefaultViewConverter(T.shape[1:]))
