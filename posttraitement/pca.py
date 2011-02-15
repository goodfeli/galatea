import os, cPickle
import numpy
from scipy import linalg

class PCA:
    """
    Layer which transforms its input via Principal Component Analysis.
    """

    def __init__(self):
        pass

    # This would eventually be the 'self.fit' mentioned here: http://bit.ly/gxSt6b
    def compute(self, X, num_components = numpy.inf, min_variance = .0):
        """
        Compute PCA transformation, but do nothing with it.

        Should only be called once.

        Given a rectangular matrix X = USV such that S is a diagonal matrix with
        X's singular values along its diagonal, computes and stores U.

        :type X: numpy.ndarray, shape (n, d)
        :param X: matrix on which to compute PCA

        :type num_components: int
        :param num_components: this many components will be preserved, in
            decreasing order of variance

        :type min_variance: float
        :param min_variance: components with variance below this threshold will be discarded
        """

        assert X.shape[1] <= X.shape[0]

        X -= numpy.mean (X, axis = 0)
        (v, self.U) = linalg.eig(numpy.cov(X.T))

        order = numpy.argsort(-v)
        v, self.U = v[order], self.U[:,order]
        var_cutoff = min(numpy.where(((v / sum(v)) < min_variance)))
        num_components = min(num_components, var_cutoff, X.shape[1])
        self.U = self.U[:,:num_components]

    # This would eventually be the 'self.output' mentioned here: http://bit.ly/gxSt6b
    def transform(self, X):
        """
        Compute and return the previously computed PCA transformation of the
        given data.

        :type X: numpy.ndarray, shape (n, d)
        :param X: matrix to transform
        """

        assert self.U is not None
        return numpy.dot(X, self.U)

    def save(self, save_dir, save_filename = 'model_pca.pkl'):
        """
        Save the computed PCA transformation matrix.
        """

        print '... saving model'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        save_file = open(os.path.join(save_dir, save_filename), 'wb')
        cPickle.dump(self.U, save_file, -1)
        save_file.close()

    def load(self, load_dir, load_filename = 'model_pca.pkl'):
        """
        Load a PCA transformation matrix.
        """

        print '... loading model'
        save_file = open(os.path.join(load_dir, load_filename), 'r')
        self.U = cPickle.load(save_file)
        save_file.close()


if __name__ == "__main__":
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
                        help = "Only the 'n' most important components will be preserved")
    parser.add_argument('-v', '--min-variance', action = 'store',
                        type = float,
                        default = .0,
                        required = False,
                        help = "Components with variance below this threshold will be discarded")
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
    pca = PCA()
    pca.compute(train_rep, args.num_components, args.min_variance)
    pca.save(args.save_dir)
    pca.load(args.save_dir)
    valid_pca = pca.transform(valid_rep)
    test_pca = pca.transform(test_rep)
    print >> stderr, "New shapes:", map(numpy.shape, [valid_pca, test_pca])
    
    # This is probably not very useful; I load this dump from R for analysis.
    print "... dumping new representation"
    map(lambda((f, d)): numpy.savetxt(f, d), zip(map (lambda(s): s + "_pca.csv", ["valid", "test"]), [valid_pca, test_pca]))
