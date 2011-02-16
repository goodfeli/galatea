import os, cPickle
import numpy
from scipy import linalg

from framework.base import Block, Trainer

class PCATrainer(Trainer):
    """
    Compute a PCA transformation matrix from the given data.
    """
    def __init__(self, inputs, **kwargs):
        """
        :type inputs: numpy.ndarray, shape (n, d)
        :param inputs: matrix from which to compute PCA transformation

        :type num_components: int
        :param num_components: this many components will be preserved, in
            decreasing order of variance

        :type min_variance: float
        :param min_variance: components with normalized variance [0-1] below 
            this threshold will be discarded
        """

        kwargs.setdefault('num_components', numpy.inf)
        kwargs.setdefault('min_variance', .0)

        super(PCATrainer, self).__init__(inputs, **kwargs)

    def updates(self):
        """
        Compute the PCA transformation matrix.

        Should only be called once.

        Given a rectangular matrix X = USV such that S is a diagonal matrix with
        X's singular values along its diagonal, computes and stores W = V^-1.

        """

        X = self.inputs

        assert "W" not in self.__dict__, "PCATrainer.updates should only be" \
            " called once"
        assert X.shape[1] <= X.shape[0], "Number of samples (rows) must be" \
            " greater than number of features (columns)"
        # Actually, I don't think is necessary, but in practice all our datasets
        # fulfill this requirement anyway, so this serves as a sanity check.

        X -= numpy.mean (X, axis = 0)
        (v, self.W) = linalg.eig(numpy.cov(X.T))

        order = numpy.argsort(-v)
        v, self.W = v[order], self.W[:,order]
        var_cutoff = min(numpy.where(((v / sum(v)) < self.min_variance)))
        num_components = min(self.num_components, var_cutoff, X.shape[1])
        self.W = self.W[:,:num_components]

        self._params = [self.W]

    def save(self, save_dir, save_filename = 'model_pca.pkl'):
        """
        Save the computed PCA transformation matrix.
        """

        print '... saving model'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        save_file = open(os.path.join(save_dir, save_filename), 'wb')
        for param in self._params:
            cPickle.dump(param, save_file, -1)
        save_file.close()

class PCA(Block):
    """
    Block which transforms its input via Principal Component Analysis.
    """

    def __init__(self, inputs):
        """
        :type inputs: numpy.ndarray, shape (n, d)
        :param inputs: matrix on which to compute PCA
        """

        super(PCA, self).__init__(inputs)

    def outputs(self):
        """
        Compute and return the PCA transformation of the current data.
        """

        X = self.inputs
        assert "W" in self.__dict__ and self.W is not None, "PCA transformation" \
            " matrix 'W' not defined"
        assert X.shape[1] == self.W.shape[0], "Incompatible input matrix shape"

        return numpy.dot(X, self.W)

    def load(self, load_dir, load_filename = 'model_pca.pkl'):
        """
        Load a PCA transformation matrix.
        """

        print '... loading model'
        load_file = open(os.path.join(load_dir, load_filename), 'r')
        self.W = cPickle.load(load_file)
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
    parser.add_argument('-u', '--dump', action='store_const',
                        default=False,
                        const=True,
                        required=False,
                        help='Dump transformed data in CSV format')
    args = parser.parse_args()

    # Load model
    da = dA()
    da.load(args.load_dir)

    # Load dataset
    data = load_data(args.dataset)
    print >> stderr, "Dataset shapes:", map(lambda(x): get_constant(x.shape), data)

    # Compute dataset representation from model
    def get_subset_rep (index):
        d = theano.tensor.matrix('input')
        return theano.function([], da.get_hidden_values(d), givens = {d:data[index]})()
    [train_rep, valid_rep, test_rep] = map(get_subset_rep, range(3))

    # Compute PCA transformation on training subset, then save and reload it
    # for no reason, then transform test and valid subsets
    print "... computing PCA"
    trainer = PCATrainer(train_rep, num_components = args.num_components,
        min_variance = args.min_variance)
    trainer.updates()
    trainer.save(args.save_dir)

    pca = PCA(valid_rep)
    pca.load(args.save_dir)
    valid_pca = pca.outputs()

    pca = PCA(test_rep)
    pca.load(args.save_dir)
    test_pca = pca.outputs()
    print >> stderr, "New shapes:", map(numpy.shape, [valid_pca, test_pca])
    
    # This is probably not very useful; I load this dump from R for analysis.
    if args.dump:
        print "... dumping new representation"
        map(lambda((f, d)): numpy.savetxt(f, d), zip(map (lambda(s): s + "_pca.csv", ["valid", "test"]), [valid_pca, test_pca]))
