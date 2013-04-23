import numpy as np
import warnings

from theano.gof.graph import Variable
from theano.sandbox.rng_mrg import MRG_RandomStreams

from pylearn2.config import yaml_parse
from pylearn2.datasets import Dataset
from pylearn2.utils import function

class S3C_Dataset(Dataset):
    """
    A dataset that is defined by an S3C model and gives an infinite stream of
    samples of v as inputs and samples of (h, s) as targets.
    """

    def __init__(self, s3c, flatten=False, include_s=True):
        self.__dict__.update(locals())
        del self.self

    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=False, rng=None):

        if mode != 'random_uniform':
            raise NotImplementedError()

        return Iterator(self, batch_size, num_batches, topo, targets, rng)

    def has_targets(self):
        return True

    def get_weights_view(self, W):
        if not hasattr(self, 'orig_dataset'):
            self.orig_dataset = yaml_parse.load(self.s3c.dataset_yaml_src)
        return self.orig_dataset.get_weights_view(W)

class Iterator(object):

    stochastic = True

    def __iter__(self):
        return self

    def __init__(self, dataset, batch_size, num_batches, topo, targets, rng):


        if rng is None:
            rng = np.random.RandomState([2013, 4, 22])
        if isinstance(rng, list):
            rng = np.random.RandomState(rng)

        self.__dict__.update(locals())
        del self.self

        theano_rng = MRG_RandomStreams(rng.randint(2 ** 16))

        if batch_size is None:
            raise ValueError("must specify batch size, there is infinite data.")

        samples = dataset.s3c.random_design_matrix(batch_size, theano_rng = theano_rng,
                            return_all = targets)
        assert samples is not None
        if targets:
            assert len(samples) == 3
            assert not any(sample is None for sample in samples)
        else:
            assert isinstance(samples, Variable)

        warnings.warn("This is recompiled every time we make a new iterator, just compile it once per iteration mode. Keep in mind the rng is part of the mode though-- the monitor wants to see the same stuff every time.")
        self.f = function([], samples)

        if num_batches is None:
            raise ValueError("must specify a number of batches, there is infinite 'data'")

        self.num_examples = num_batches * batch_size

    def next(self):
        if self.num_batches is 0:
            raise StopIteration()

        if self.topo:
            raise NotImplementedError()

        samples = self.f()

        if self.targets:
            Y = (samples[0], samples[1])
            if self.dataset.include_s:
                if self.dataset.flatten:
                    Y = np.concatenate(Y, axis=1)
            else:
                Y = Y[0]
            rval = (samples[2], Y)
        else:
            rval = samples

        self.num_batches-= 1

        return rval
