import numpy as np

from theano import tensor as T

from pylearn2.models.mlp import Layer
from pylearn2.models.maxout import Maxout

def shuffle(dataset, seed = None):

    if seed is None:
        seed = [2013, 2, 26]

    rng = np.random.RandomState(seed)

    for i in xrange(dataset.X.shape[0]):
        j = rng.randint(dataset.X.shape[0])
        tmp = dataset.X[i,:].copy()
        dataset.X[i,:] = dataset.X[j,:].copy()
        dataset.X[j,:] = tmp
        tmp = dataset.y[i,:].copy()
        dataset.y[i,:] = dataset.y[j,:].copy()
        dataset.y[j,:] = tmp

    return dataset

class GCN_C01B(Layer):
    """
    Like pylearn2.datasets.preprocessing.GlobalContrastNormalization but
        -uses theano expressions instead of numpy calls, thus enabling it to be
        -written as an MLP Layer rather than a Preprocessor so it can work on the fly
        -written to work on the topological view so it doesn't disrupt convolution pipelines
    """

    def __init__(self, layer_name, subtract_mean=True, std_bias=10.0, use_norm=False):
        """

        Optionally subtracts the mean of each example
        Then divides each example either by the standard deviation of the
        pixels contained in that example or by the norm of that example

        Parameters:

            subtract_mean: boolean, if True subtract the mean of each example
            std_bias: Add this amount inside the square root when computing
                      the standard deviation or the norm
            use_norm: If True uses the norm instead of the standard deviation


            The default parameters of subtract_mean = True, std_bias = 10.0,
            use_norm = False are used in replicating one step of the
            preprocessing used by Coates, Lee and Ng on CIFAR10 in their paper
            "An Analysis of Single Layer Networks in Unsupervised Feature
            Learning"
        """

        self.layer_name = layer_name
        self.subtract_mean = subtract_mean
        self.std_bias = std_bias
        self.use_norm = use_norm
        self._params = []

    def set_input_space(self, space):
        assert tuple(space.axes) == ('c', 0, 1, 'b')
        self.input_space = space
        self.output_space = space

    def fprop(self, state_below):
        c01b = state_below
        self.input_space.validate(c01b)

        if self.subtract_mean:
            c01b = c01b - c01b.mean(axis=(0,1,2)).dimshuffle('x','x','x',0)

        if self.use_norm:
            scale = T.sqrt(T.square(c01b).sum(axis=(0,1,2)) + self.std_bias)
        else:
            # use standard deviation
            scale = T.sqrt(T.square(c01b).mean(axis=(0,1,2)) + self.std_bias)
        eps = 1e-8
        scale = (scale < eps) + (scale >= eps) * scale

        c01b = c01b / scale.dimshuffle('x', 'x', 'x', 0)

        return c01b

from pylearn2.train_extensions import TrainExtension
from pylearn2.gui.patch_viewer import PatchViewer
class DatasetRecorderDisplayer(TrainExtension):

    def __init__(self, datasets):
        self.__dict__.update(locals())
        del self.self

    def on_monitor(self, *args, **kwargs):

        if not hasattr(self, 'record'):
            self.record = {}
            self.size = {}
            for dataset in self.datasets:
                assert tuple(dataset.view_converter.axes) == ('c', 0, 1, 'b')
                self.record[dataset] = dataset.get_topological_view().copy()
                self.size[dataset] = dataset.X.shape[0]
        else:
            for i, dataset in enumerate(self.datasets):
                size = self.size[dataset]
                assert dataset.X.shape[0] == size
                self.record[dataset] = np.concatenate((self.record[dataset], dataset.get_topological_view().copy()),
                        axis=-1)
                record_view = self.record[dataset].copy()
                record_view /= np.abs(record_view).max()
                pv = PatchViewer(grid_shape=(record_view.shape[3]/size, size),
                        patch_shape = record_view.shape[1:3], is_color = record_view.shape[0] == 3)
                for j in xrange(record_view.shape[3]):
                    pv.add_patch(np.transpose(record_view[:,:,:,j], (1, 2, 0)), rescale=False)
                print 'Dataset %d: ' % i
                pv.show()
                x = raw_input()

def pad(dataset, amt):

    axes = dataset.view_converter.axes
    if not( tuple(axes) == ('c', 0, 1, 'b')):
        print axes
        assert False
    t = dataset.get_topological_view()
    padded = np.zeros((t.shape[0], t.shape[1] + amt, t.shape[2] + amt, t.shape[-1]), dtype='float32')
    padded[:,amt/2:amt/2+t.shape[1],amt/2:amt/2+t.shape[2],:] = t
    dataset.set_topological_view(padded, axes)

    return dataset

class Sphere(Maxout):

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below) + self.b

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if not hasattr(self, 'pool_stride'):
            self.pool_stride = self.pool_size

        if self.randomize_pools:
            z = T.dot(z, self.permute)

        if not hasattr(self, 'min_zero'):
            self.min_zero = False

        if self.min_zero:
            p = T.zeros_like(z)
        else:
            p = None

        last_start = self.detector_layer_dim  - self.pool_size
        for i in xrange(self.pool_size):
            cur = z[:,i:last_start+i+1:self.pool_stride]
            if p is None:
                p = cur
            else:
                p = T.maximum(cur, p)

        p.name = self.layer_name + '_p_'

        norms = T.sqrt(T.sqr(p).sum(axis=1))

        p = p / norms.dimshuffle(0, 'x')

        return p

class SphereGroups(Maxout):

    def __init__(self, group_size, ** kwargs):
        self.group_size = group_size
        super(SphereGroups, self).__init__(**kwargs)

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below) + self.b

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if not hasattr(self, 'pool_stride'):
            self.pool_stride = self.pool_size

        if self.randomize_pools:
            z = T.dot(z, self.permute)

        if not hasattr(self, 'min_zero'):
            self.min_zero = False

        if self.min_zero:
            p = T.zeros_like(z)
        else:
            p = None

        last_start = self.detector_layer_dim  - self.pool_size
        for i in xrange(self.pool_size):
            cur = z[:,i:last_start+i+1:self.pool_stride]
            if p is None:
                p = cur
            else:
                p = T.maximum(cur, p)

        p.name = self.layer_name + '_p_'


        p = p.reshape((p.shape[0], p.shape[1] / self.group_size, self.group_size))

        norms = T.sqrt(T.sqr(p).sum(axis=2))

        p = p / norms.dimshuffle(0, 1, 'x')

        p = p.reshape((p.shape[0], p.shape[1] * self.group_size))

        return p


class Universal(Maxout):

    def __init__(self, ** kwargs):
        kwargs['num_units'] = 2
        super(Universal, self).__init__(**kwargs)

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below) + self.b

        p = T.concatenate(
                (z[:,:z.shape[1]/2].max(axis=1).dimshuffle(0, 'x'),
                z[:z.shape[1]/2:].max(axis=1).dimshuffle(0, 'x')),
                axis=1)

        return p
