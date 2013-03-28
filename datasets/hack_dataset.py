from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import numpy as np

class HackDataset(DenseDesignMatrix):

    def __init__(self, labels_from, X, start, stop):

        super(HackDataset, self).__init__(X = X, y = labels_from.y)

        convert_to_one_hot = True
        if convert_to_one_hot:
            if not ( self.y.min() == 0):
                raise AssertionError("Expected y.min == 0 but y.min == "+str(self.y.min()))
            nclass = self.y.max() + 1
            y = np.zeros((self.y.shape[0], nclass), dtype='float32')
            for i in xrange(self.y.shape[0]):
                y[i,self.y[i]] = 1.
            self.y = y

        self.X = self.X[start:stop,:]
        assert self.X.shape[0] == stop - start
        self.y = self.y[start:stop,:]
        assert self.y.shape[0] == stop - start

