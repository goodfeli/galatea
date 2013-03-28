from pylearn2.utils.image import make_letterboxed_thumbnail
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.utils import image


import numpy as np

class DARPA_ImageNet(DenseDesignMatrix):
    def __init__(self, iterator, num_examples, image_shape):
        assert len(image_shape) == 2

        T = np.zeros((num_examples,image_shape[0],image_shape[1],3),dtype='float32')

        for i in xrange(num_examples):
            image_path = iterator.next()
            img = image.load(image_path)
            T[i,:] = make_letterboxed_thumbnail(img, image_shape)

        super(DARPA_ImageNet, self).__init__(
                topo_view = T,
                view_converter = DefaultViewConverter(T.shape[1:]))
