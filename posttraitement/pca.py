import os, cPickle

import numpy
from scipy import linalg
import theano
from theano import tensor as T

from framework.base import Block, Optimizer
from framework.utils import sharedX


class PCA(Block):
    """
    Block which transforms its input via Principal Component Analysis.
    """

    def __init__(self, conf):
        """
        Parameters in conf:

        :type num_components: int
        :param num_components: this many components will be preserved, in
            decreasing order of variance

        :type min_variance: float
        :param min_variance: components with normalized variance [0-1] below
            this threshold will be discarded

        :type whiten: bool
        :param whiten: whether or not to divide projected features by their
            standard deviation
        """

        self.num_components = conf.get('num_components', numpy.inf)
        self.min_variance = conf.get('min_variance', 0.0)
        self.whiten = conf.get('whiten', False)

        # Initialize self.W with an empty matrix
        # self.W is set only so that self._params can be built up-to-date.
        # If we figure out another solution, why not.
        self.W = sharedX(
            numpy.zeros((0, 0)),
            name='W',
            borrow=True
        )

        self.v = sharedX(
            numpy.zeros((0)),
            name='v',
            borrow=True
        )

        self.mean = sharedX(
            numpy.zeros((0)),
            name='mean',
            borrow=True
        )

        # This module really has no adjustable parameters -- once train()
        # is called once, they are frozen, and are not modified via gradient
        # descent.
        self._params = []

    def train(self, X):
        """
        Compute the PCA transformation matrix.

        Should only be called once.

        Given a rectangular matrix X = USV such that S is a diagonal matrix with
        X's singular values along its diagonal, computes and returns W = V^-1.
        """

        # Actually, I don't think is necessary, but in practice all our datasets
        # fulfill this requirement anyway, so this serves as a sanity check.
        # TODO: Implement the snapshot method for the p >> n case.
        assert X.shape[1] <= X.shape[0], "Number of samples (rows) must be" \
            " greater than number of features (columns)"
        # Implicit copy done below.
        mean = numpy.mean(X, axis=0)
        X = X - mean
        # The following computation is always carried in double precision
        v, W = linalg.eig(numpy.cov(X.T))
        order = numpy.argsort(-v)
        v, W = v[order], W[:, order]
        var_cutoff = min(numpy.where(((v / v.sum()) < self.min_variance)))
        num_components = min(self.num_components, var_cutoff, X.shape[1])
        v, W = v[:num_components], W[:,:num_components]

        # Update Theano shared variable
        W = theano._asarray(W, dtype = theano.config.floatX)
        v = theano._asarray(v, dtype = theano.config.floatX)
        mean = theano._asarray(mean, dtype = theano.config.floatX)
        self.W.set_value(W, borrow = True)
        if self.whiten:
            self.v.set_value(v, borrow = True)
        self.mean.set_value(mean, borrow = True)

    def __call__(self, inputs):
        """
        Compute and return the PCA transformation of the current data.

        :type inputs: numpy.ndarray, shape (n, d)
        :param inputs: matrix on which to compute PCA
        """

        #assert "W" in self.__dict__ and self.W.get_value(borrow=True).shape[0] > 0,\
        #        "PCA transformation matrix 'W' not defined"
        #assert inputs.get_value().shape[1] == self.W.get_value().shape[0], \
        #    "Incompatible input matrix shape"

        Y = T.dot(inputs - self.mean, self.W)
        # If eigenvalues are defined, self.whiten was True.
        if numpy.any(self.v.get_value() > 0):
            Y /= T.sqrt(self.v)
        return Y

    def save(self, save_dir, save_filename = 'model_pca.pkl'):
        """
        Save the computed PCA transformation matrix.
        """

        print '... saving model'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        save_file = open(os.path.join(save_dir, save_filename), 'wb')
        cPickle.dump(self.W.get_value(), save_file, -1)
        cPickle.dump(self.v.get_value(), save_file, -1)
        cPickle.dump(self.mean.get_value(), save_file, -1)
        save_file.close()

    def load(self, load_dir, load_filename = 'model_pca.pkl'):
        """
        Load a PCA transformation matrix.
        """

        print '... loading model'
        load_file = open(os.path.join(load_dir, load_filename), 'r')
        self.W.set_value(cPickle.load(load_file))
        self.v.set_value(cPickle.load(load_file))
        self.mean.set_value(cPickle.load(load_file))
        load_file.close()

if __name__ == "__main__":
    """
    Run a dataset through a previously learned dA model, compute a PCA
    transformation matrix from the training subset, pickle it, then apply said
    transformation to the test and valid subsets and dump these representations.
    """

    from sys import stderr
    import argparse
    from dense.dA import dA
    from dense.logistic_sgd import load_data, get_constant
    import theano

    parser = argparse.ArgumentParser(
        description="Transform the output of a model by Principal Component Analysis"
    )
    parser.add_argument('dataset', action = 'store',
                        type = str,
                        choices = ['avicenna', 'harry', 'rita', 'sylvester',
                                 'ule'],
                        help = 'Dataset on which to run the PCA')
    parser.add_argument('-d', '--load-dir', action = 'store',
                        type = str,
                        default = ".",
                        required = False,
                        help = "Directory from which to load original model.pkl")
    parser.add_argument('-s', '--save-dir', action = 'store',
                        type = str,
                        default = ".",
                        required = False,
                        help = "Directory where model pickle is to be saved")
    parser.add_argument('-n', '--num-components', action = 'store',
                        type = int,
                        default = numpy.inf,
                        required = False,
                        help = "Only the 'n' most important components will be"
                            " preserved")
    parser.add_argument('-v', '--min-variance', action = 'store',
                        type = float,
                        default = .0,
                        required = False,
                        help = "Components with variance below this threshold"
                            " will be discarded")
    parser.add_argument('-w', '--whiten', action='store_const',
                        default=False,
                        const=True,
                        required=False,
                        help='Divide projected features by their standard deviation')
    parser.add_argument('-u', '--dump', action='store_const',
                        default=False,
                        const=True,
                        required=False,
                        help='Dump transformed data in CSV format')
    args = parser.parse_args()

    # Load (non-framework) dA model.
    da = dA()
    da.load(args.load_dir)

    # Load dataset.
    data = load_data(args.dataset)
    print >> stderr, "Dataset shapes:", map(lambda(x): get_constant(x.shape), data)

    # Compute dataset representation from model.
    def get_subset_rep (index):
        d = T.matrix('input')
        return theano.function([], da.get_hidden_values(d), givens = {d:data[index]})()
    [train_rep, valid_rep, test_rep] = map(get_subset_rep, range(3))

    # Compute PCA transformation on training subset, then save and reload it
    # for no reason, then transform test and valid subsets.
    print "... computing PCA"

    # First, build a config. dict.
    conf = {
        'num_components': args.num_components,
        'min_variance': args.min_variance,
        'whiten': args.whiten
    }

    # A symbolic input representing the data.
    inputs = T.dmatrix()

    # Allocate a PCA block.
    pca = PCA(conf)

    # Compute the PCA transformation matrix from the training data.
    pca.train(train_rep)

    # Save transformation matrix to pickle, then reload it.
    #pca.save(args.save_dir)
    #pca.load(args.save_dir)

    # Apply the transformation to test and valid subsets.
    pca_transform = theano.function([inputs], pca(inputs))
    valid_pca = pca_transform(valid_rep)
    test_pca = pca_transform(test_rep)

    print >> stderr, "New shapes:", map(numpy.shape, [valid_pca, test_pca])
    
    # This is probably not very useful; I load this dump from R for analysis.
    if args.dump:
        print "... dumping new representation"
        map(lambda((f, d)): numpy.savetxt(f, d), zip(map (lambda(s): s + "_pca.csv",
            ["valid", "test"]), [valid_pca, test_pca]))
