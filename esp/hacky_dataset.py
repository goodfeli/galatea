from pylearn2.datasets.dataset import Dataset
import numpy as np
from pylearn2.utils import serial

class Dummy(object):
    axes = ['c', 0, 1, 'b']

class HackyDataset(Dataset):

    directory = '/data/lisatmp/goodfeli/hacky_dataset'

    # view_converter = Dummy()

    def get_topo_batch_axis(self):
        return 3

    def __init__(self, start_batch = 0, stop_batch = 782):
        self.__dict__.update(locals())
        del self.self
        self.bow = serial.load('/data/lisatmp/goodfeli/esp_bow.pkl')
        self.global_rng = np.random.RandomState([2013, 3, 28])
        self.y = self.bow.X[start_batch*128:stop_batch*128,:]

    def get_batch_topo(self, batch_size):
        assert batch_size == 128

        idx = self.global_rng.randint(self.stop_batch - self.start_batch)

        rval = np.load(self.directory + '/%d.npy' % idx)

        print rval.shape

        return rval

    def iterator(self,
        mode=None, batch_size=None, num_batches=None,
                         topo=None, targets=False, rng=None):
        assert mode == 'sequential'
        assert batch_size == 128
        assert num_batches is None
        assert topo
        assert rng is None
        return Iterator(self, targets)

    def has_targets(self):
        return True

    def num_examples(self):
        if hasattr(self,'m'):
            return self.m
        m = 0
        for batch in Iterator(self, False):
            m += batch.shape[-1]
        self.m =m
        return m

class Iterator(object):

    stochastic = False

    def __iter__(self):
        return self

    def __init__(self, dataset, targets):
        self.targets = targets
        self.dataset = dataset
        self.pos = dataset.start_batch

    def next(self):

        print 'iterator: ',self.pos
        if self.pos == self.dataset.stop_batch:
            raise StopIteration()

        rval = np.load(self.dataset.directory + '/%d.npy' % self.pos)

        if self.targets:
            rval = (rval, self.dataset.bow.X[self.pos*128:(self.pos+1)*128,:])

        self.pos += 1

        return rval

    def num_examples_getter(self):
        return self.dataset.num_examples()

    def num_examples_setter(self, v):
        assert False

    def num_examples_deleter(self):
        assert False

    num_examples = property(num_examples_getter, num_examples_setter, num_examples_deleter)
