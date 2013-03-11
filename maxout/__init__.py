import numpy as np

from theano import tensor as T

from pylearn2.models.mlp import Layer

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
