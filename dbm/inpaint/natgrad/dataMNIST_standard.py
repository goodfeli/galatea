import theano
import numpy
import time


class DataMNIST(object):
    def __init__(self, path, mbs, bs, rng, same_batch, callback):
        self.path = path
        self.mbs = mbs
        self.bs = bs
        self.rng = rng
        self.callback = callback
        self.same_batch = same_batch
        if same_batch:
            assert mbs == bs
        self.data = numpy.load(path)
        self.xdim = self.data['train_x'].shape[1]
        self.ydim = numpy.max(self.data['train_y'])+1

        self.data_x = self.data['train_x']
        self.data_y = numpy.zeros((50000, self.ydim), dtype='float32')
        self.data_y[numpy.arange(50000), self.data['train_y']] = 1.

        self.n_batches = 50000 // self.bs
        self.nat_batches = 50000 // self.mbs
        self.grad_perm = self.rng.permutation(self.n_batches)
        self.nat_perm = self.rng.permutation(self.nat_batches)
        self.pos = -1
        self.nat_pos = -1
        self._train_x = theano.shared(numpy.zeros((bs, self.xdim),
                                                  dtype='float32'),
                                      'train_x')
        self._train_y = theano.shared(numpy.zeros((bs, self.ydim),
                                                  dtype='float32'),
                                      'train_y')

        if not same_batch:
            self._natgrad = theano.shared(numpy.zeros((mbs, self.xdim),
                                                      dtype='float32'),
                                          'natgrad_x')
            self._natgrady = theano.shared(numpy.zeros((mbs, self.ydim),
                                                       dtype='float32'),
                                           'natgrad_y')
        else:
            self._natgrad = self._train_x
            self._natgrady = self._train_y
        self.variables = [self._train_x, self._train_y]



    def update_before_computing_gradients(self):
        self.pos = (self.pos + 1) % self.n_batches
        if self.pos % self.n_batches == 0:
            self.grad_perm = self.rng.permutation(self.n_batches)
        offset = self.grad_perm[self.pos]
        begin = offset * self.bs
        end = (offset+1)*self.bs
        dX = self.data_x[begin:end]
        dY = self.data_y[begin:end]
        self.callback(dX, dY)
        self._train_x.set_value(dX, borrow=True)
        self._train_y.set_value(dY, borrow=True)

    def update_before_computing_natural_gradients(self):
        if not self.same_batch:
            # What to do with the callback
            self.nat_pos = (self.nat_pos + 1) % self.nat_batches
            if self.nat_pos % self.nat_batches == 0:
                self.nat_perm = self.rng.permutation(self.nat_batches)
            offset = self.nat_perm[self.nat_pos]
            begin = offset * self.mbs
            end = (offset+1)*self.mbs
            return self.train_x[begin:end]

    def update_before_evaluation(self):
        pass
