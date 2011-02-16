"""Script for running experiments."""
# -*- coding: latin-1 -*-
# Standard library imports
import copy
import cPickle
import gzip
import os
import sys
import time

# Third-party lbirary imports
import argparse
import numpy
import theano
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from logistic_sgd import load_data, get_constant
from posttraitement import pca
from utils import tile_raster_images

import PIL.Image


class dA(object):
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """

    def __init__(self,
                 seed_params = None,
                 seed_noise = None,
                 input = None,
                 n_visible= 784,
                 n_hidden= 500,
                 tied_weights = True,
                 act_enc = 'sigmoid',
                 act_dec = 'sigmoid',
                 W = None,
                 W_prime = None,
                 b = None,
                 b_prime = None):

        # using cPickle serialization
        self.__initargs__ = copy.copy(locals())
        del self.__initargs__['self']

        self.n_visible = n_visible
        self.n_hidden  = n_hidden
        self.tied_weights = tied_weights

        assert act_enc in set(['sigmoid', 'tanh'])
        assert act_dec in set(['sigmoid', 'softplus', 'linear'])
        self.act_enc = act_enc
        self.act_dec = act_dec

        # create a numpy random generator
        # this is used for the initialization of the parameters
        if not seed_params:
            numpy_rng        = numpy.random.RandomState(123)
            self.seed_params = 123
        else:
            self.seed_params = seed_params
            numpy_rng        = numpy.random.RandomState(seed_params)
        self.numpy_rng = numpy_rng

        # create a Theano random generator that gives symbolic random values
        # this is used for adding noise to the input
        if not seed_noise:
            theano_rng = RandomStreams(self.numpy_rng.randint(2**30))
            self.seed_noise = 2**30
        else:
            self.seed_noise = seed_noise
            theano_rng = RandomStreams(self.numpy_rng.randint(seed_noise))
        self.theano_rng = theano_rng

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(self.numpy_rng.uniform(
                low  = -4*numpy.sqrt(6./(n_hidden+n_visible)),
                high =  4*numpy.sqrt(6./(n_hidden+n_visible)),
                size = (n_visible, n_hidden)), dtype = theano.config.floatX)
            W = theano.shared(value = initial_W, name ='W')

        if not b_prime:
            b_prime = theano.shared(value = numpy.zeros(n_visible,
                dtype = theano.config.floatX))

        if not b:
            b = theano.shared(value = numpy.zeros(n_hidden,
                dtype = theano.config.floatX), name ='b')


        self.W = W
        # b corresponds to the bias of the hidden
        self.b = b
        # b_prime corresponds to the bias of the visible
        self.b_prime = b_prime

        if self.tied_weights:
            self.W_prime = self.W.T
        else:
            if not W_prime:
                # not sure about the initialization
                initial_W_prime = numpy.asarray(self.numpy_rng.uniform(
                    low  = -4*numpy.sqrt(6./(n_hidden+n_visible)),
                    high =  4*numpy.sqrt(6./(n_hidden+n_visible)),
                    size = (n_hidden, n_visible)), dtype = theano.config.floatX)
                W_prime = theano.shared(value = initial_W_prime, name ='W_prime')

            self.W_prime = W_prime

        # if no input is given, generate a variable representing the input
        if input == None :
            # we use a matrix because we expect a minibatch of several examples,
            # each example being a row
            self.x = T.dmatrix(name = 'input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

        if not self.tied_weights:
            self.params.append(self.W_prime)

        self.output = self.get_hidden_values(self.x)

    def get_corrupted_input(self, input, corruption_level, noise='binomial'):
        """ This function keeps ``1-corruption_level`` entries of the inputs the same
        and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a probability of
                1 - ``corruption_level`` and 0 with ``corruption_level``

                The binomial function return int64 data type by default.
                int64 multiplicated by the input type(floatX) always return float64.
                To keep all data in floatX when floatX is float32, we set the dtype
                of the binomial to floatX. As in our case the value of the binomial
                is always 0 or 1, this don't change the result. This is needed to allow
                the gpu to work correctly as it only support float32 for now.
        """
        if noise == 'binomial':
            return self.theano_rng.binomial(size = input.shape, n = 1,
                p = 1 - corruption_level, dtype=theano.config.floatX) * input
        elif noise == 'gaussian':
            return input + self.theano_rng.normal(size = input.shape, avg = 0,
                std = corruption_level, dtype = theano.config.floatX)
        else:
            raise NotImplementedError('This noise %s is not implemented yet'%(noise))

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        if self.act_enc == 'sigmoid':
            return T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        elif self.act_enc == 'tanh':
            return T.tanh(T.dot(input, self.W) + self.b)
        else:
            raise NotImplementedError('Encoder function %s is not implemented yet' \
                %(self.act_enc))

    def get_reconstructed_input(self, hidden ):
        """ Computes the reconstructed input given the values of the hidden layer """
        if self.act_dec == 'sigmoid':
            return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
        elif self.act_dec == 'linear':
            return T.dot(hidden, self.W_prime) + self.b_prime
        elif self.act_dec == 'softplus':
            def softplus(x):
                return T.log(1. + T.exp(x))
            return softplus(T.dot(hidden, self.W_prime) + self.b_prime)
        else:
            raise NotImplementedError('Decoder function %s is not implemented yet' \
                %(self.act_dec))

    def get_cost_updates(self, corruption_level, learning_rate, cost = 'CE',
        noise = 'binomial'):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level, noise)
        y       = self.get_hidden_values( tilde_x)
        z       = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using minibatches,
        #        L will  be a vector, with one entry per example in minibatch
        if cost == 'CE':
            L = - T.sum( self.x*T.log(z) + (1-self.x)*T.log(1-z), axis=1 )
        elif cost == 'MSE':
            L = T.sum( (self.x-z)**2, axis=1 )
        else:
            raise NotImplementedError('This cost function %s is not implemented yet' \
                %(cost))

        # note : L is now a vector, where each element is the cross-entropy cost
        #        of the reconstruction of the corresponding example of the
        #        minibatch. We need to compute the average of all these to get
        #        the cost of the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param -  learning_rate*gparam

        return (cost, updates)

    def fit(self, dataset, learning_rate, batch_size=20, epochs=50, cost='CE',
        noise='gaussian', corruption_level=0.3):
        """ This function fits the dA to the dataset given
        some hyper-parameters and returns the loss evolution
        and the time spent during training   """


        # compute number of minibatches for training, validation and testing
        n_train_batches = get_constant(dataset.shape[0]) / batch_size

        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch

        cost, updates = self.get_cost_updates(corruption_level = corruption_level,
            learning_rate = learning_rate,
            noise = noise,
            cost = cost)
        train_da = theano.function([index],
            cost,
            updates = updates,
            givens = {self.x:dataset[index*batch_size:(index+1)*batch_size]},
            name ='train_da'
            )

        start_time = time.clock()

        ############
        # TRAINING #
        ############
        loss = []
        print '... training model...'
        # go through training epochs


        for epoch in xrange(epochs):
            tic = time.clock()
            # go through trainng set
            c = []
            for batch_index in xrange(int(n_train_batches)):
                c.append(train_da(batch_index))

            toc = time.clock()
            loss.append(numpy.mean(c))
            print 'Training epoch %d, time spent (min) %f,  cost ' \
                %(epoch,(toc-tic)/60.), numpy.mean(c)
            toc = tic

        end_time = time.clock()
        training_time = (end_time - start_time)/60.

        return training_time, loss

    def save(self, save_dir, save_filename = 'model.pkl'):
        """ save the parameters of the model """
        print '... saving model'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        save_file = open(os.path.join(save_dir, save_filename),'wb')
        cPickle.dump(self.__initargs__, save_file, -1)
        cPickle.dump(self.W.value, save_file, -1)
        if not self.tied_weights:
            cPickle.dump(self.W_prime.value, save_file, -1)
        cPickle.dump(self.b.value, save_file, -1)
        cPickle.dump(self.b_prime.value, save_file, -1)
        save_file.close()

    def load(self, load_dir, load_filename = 'model.pkl'):
        """ load the model """
        print '... loading model'
        save_file = open(os.path.join(load_dir, load_filename),'r')
        args = cPickle.load(save_file)
        self.__init__(
                 seed_params = args['seed_params'],
                 seed_noise = args['seed_noise'],
                 input = args['input'],
                 n_visible= args['n_visible'],
                 n_hidden= args['n_hidden'],
                 tied_weights = args['tied_weights'],
                 act_enc = args['act_enc'],
                 act_dec = args['act_dec'],
                 W = args['W'],
                 W_prime = args['W_prime'],
                 b = args['b'],
                 b_prime = args['b_prime'])

        self.W.value = cPickle.load(save_file)
        if not self.tied_weights:
            self.W_prime.value = cPickle.load(save_file)
        self.b.value = cPickle.load(save_file)
        self.b_prime.value = cPickle.load(save_file)
        save_file.close()

    def get_denoising_error(self, dataset, cost, noise, corruption_level):
        """ This function returns the denoising error over the dataset """
        batch_size = 100
        # compute number of minibatches for training, validation and testing
        n_train_batches =  get_constant(dataset.shape[0]) / batch_size

        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch

        cost, updates = self.get_cost_updates(corruption_level = corruption_level,
            learning_rate = 0.,
            noise = noise,
            cost = cost)
        get_error = theano.function([index], cost, updates = {}, givens = {
            self.x:dataset[index*batch_size:(index+1)*batch_size]},
            name='get_error')

        denoising_error = []
        # go through the dataset
        for batch_index in xrange(n_train_batches):
            denoising_error.append(get_error(batch_index))

        return numpy.mean(denoising_error)

def create_submission(dataset, save_dir_model, save_dir_submission,
    normalize_on_the_fly = False, do_pca = False):
    """
    Create submission files given the path of a model and
    a dataset.

    params:
    * dataset
        is a string corresponding to the name of the dataset
    * save_dir_model
        is the path where you saved your model
    * save_dir_submission
        is the path where you want to store the submission files
    * do_pca
        whether or not to apply (previously computed) PCA transform on model
    """
    # load the dataset
    datasets = load_data(dataset, not normalize_on_the_fly, normalize_on_the_fly)
    valid_set_x = datasets[1]
    test_set_x = datasets[1]

    # load the model
    da = dA()
    da.load(save_dir_model)

    # theano functions to get representations of the dataset learned by the model
    index = T.lscalar()    # index to a [mini]batch
    x = theano.tensor.matrix('input')

    get_rep_valid = theano.function([index], da.get_hidden_values(x), updates = {},
        givens = {x:valid_set_x},
        name = 'get_rep_valid')
    get_rep_test = theano.function([index], da.get_hidden_values(x), updates = {},
        givens = {x:test_set_x},
        name = 'get_rep_test')

    # valid and test representations
    valid_rep1 = get_rep_valid(0)
    test_rep1 = get_rep_test(0)

    if do_pca:
        pca_block = pca.PCA()
        pca_block.load(save_dir_model)
        valid_rep1 = pca_block(valid_rep1)

        pca_block = pca.PCA()
        pca_block.load(save_dir_model)
        test_rep1 = pca_block(test_rep1)

    valid_rep2 = numpy.dot(valid_rep1,valid_rep1.T)
    test_rep2 = numpy.dot(test_rep1,test_rep1.T)

    # write it in a .txt file
    valid_rep1 = numpy.floor((valid_rep1 / valid_rep1.max())*999)
    valid_rep2 = numpy.floor((valid_rep2 / valid_rep2.max())*999)

    test_rep1 = numpy.floor((test_rep1 / test_rep1.max())*999)
    test_rep2 = numpy.floor((test_rep2 / test_rep2.max())*999)

    val1 = open(save_dir_submission + dataset + '_dl_valid.prepro','w')
    val2 = open(save_dir_submission + dataset + '_sdl_valid.prepro','w')
    test1 = open(save_dir_submission + dataset + '_dl_final.prepro','w')
    test2 = open(save_dir_submission + dataset + '_sdl_final.prepro','w')

    vtxt1, ttxt1 = '', ''
    vtxt2, ttxt2 = '', ''

    for i in range(valid_rep1.shape[0]):
        for j in range(valid_rep1.shape[0]):
            vtxt2 += '%s '%int(valid_rep2[i,j])
        for j in range(valid_rep1.shape[1]):
            vtxt1 += '%s '%int(valid_rep1[i,j])
        vtxt1 += '\n'
        vtxt2 += '\n'
    del valid_rep1, valid_rep2

    for i in range(test_rep1.shape[0]):
        for j in range(test_rep1.shape[0]):
            ttxt2 += '%s '%int(test_rep2[i,j])
        for j in range(test_rep1.shape[1]):
            ttxt1 += '%s '%int(test_rep1[i,j])
        ttxt1 += '\n'
        ttxt2 += '\n'
    del test_rep1, test_rep2

    val1.write(vtxt1)
    test1.write(ttxt1)
    val2.write(vtxt2)
    test2.write(ttxt2)
    val1.close()
    test1.close()
    val2.close()
    test2.close()

    print >> sys.stderr, "... done creating files"

    os.system('zip %s %s %s'%(save_dir_submission+dataset+'_dl.zip',
        save_dir_submission+dataset+'_dl_valid.prepro',
        save_dir_submission+dataset+'_dl_final.prepro'))
    os.system('zip %s %s %s'%(save_dir_submission+dataset+'_sdl.zip',
        save_dir_submission+dataset+'_sdl_valid.prepro',
        save_dir_submission+dataset+'_sdl_final.prepro'))

    print >> sys.stderr, "... files compressed"

    os.system('rm %s %s %s %s'%(
        save_dir_submission+dataset+'_dl_valid.prepro',
        save_dir_submission+dataset+'_dl_final.prepro',
        save_dir_submission+dataset+'_sdl_valid.prepro',
        save_dir_submission+dataset+'_sdl_final.prepro'))

    print >> sys.stderr, "... useless files deleted"

def main_train(dataset, save_dir, n_hidden, tied_weights, act_enc,
    act_dec, learning_rate, batch_size, epochs, cost_type,
    noise_type, corruption_level, normalize_on_the_fly = False, do_pca = False,
    num_components = numpy.inf, min_variance = .0, do_create_submission = False,
    submission_dir = None):
    ''' main function used for training '''

    datasets = load_data(dataset, not normalize_on_the_fly, normalize_on_the_fly)
    train_set_x = datasets[0]
    valid_set_x = datasets[1]

    d = get_constant(train_set_x.shape[1])
    da = dA(n_visible = d, n_hidden = n_hidden,
            tied_weights = tied_weights,
            act_enc = act_enc, act_dec = act_dec)

    time_spent, loss = da.fit(train_set_x, learning_rate, batch_size, epochs,
        cost_type, noise_type, corruption_level)

    if save_dir:
        da.save(save_dir)

    denoising_error = da.get_denoising_error(valid_set_x, cost_type,
        noise_type, corruption_level)
    print 'Training complete in %f (min) with final denoising error %f' \
        %(time_spent,denoising_error)

    if do_pca:
        print "... computing PCA"
        x = theano.tensor.matrix('input')
        get_rep_train = theano.function([], da.get_hidden_values(x), updates = {},
            givens = {x:train_set_x}, name = 'get_rep_valid')
        pca_trainer = pca.PCATrainer(get_rep_train(), num_components = args.num_components,
            min_variance = args.min_variance)
        pca_trainer.updates()
        pca_trainer.save(args.save_dir)

    if do_create_submission:
        print "... creating submission"
        if submission_dir is None:
            submission_dir = save_dir
        create_submission(dataset, save_dir, submission_dir, normalize_on_the_fly, do_pca)

    return denoising_error, time_spent, loss



if __name__ == '__main__':

    # you can train a denoising autoencoder using this cmd:
    # python <thisfile> dataset #hidden_units tied_weights act_enc act_dec
    # costtype learning_rate batchsize epochs noise_type corruption_level
    #
    # here a few examples
    #
    # python dA.py avicenna 500 True 'sigmoid' 'linear' 'MSE' 0.01 20 50 'gaussian' 0.3
    # attention ajouter -N pour rita, qui indique que les données doivent
    # être normalisées à la volée.
    # python dA.py rita 500 True 'sigmoid' 'sigmoid' 'CE' 0.01 20 50 'gaussian' 0.3 -N
    # python dA.py sylvester 500 True 'sigmoid' 'linear' 'MSE' 0.01 20 50 'gaussian' 0.3
    # python dA.py ule 500 True 'sigmoid' 'sigmoid' 'CE' 0.01 1 50 'gaussian' 0.3
    #
    # Pour harry si l'on n'exploite pas la sparsite on peut lancer avec normalisation a
    # la volée (-N, comme rita)
    # python dA.py harry 500 True 'sigmoid' 'sigmoid' 'CE' 0.01 20 50 'gaussian' 0.3 -N
    parser = argparse.ArgumentParser(
        description='Run denoising autoencoder experiments on dense features.'
    )
    parser.add_argument('dataset', action='store',
                        type=str,
                        choices=['avicenna', 'harry', 'rita', 'sylvester',
                                 'ule'],
                        help='Dataset on which to run experiments')
    parser.add_argument('n_hidden', action='store',
                        type=int,
                        help='Number of hidden units')
    parser.add_argument('tied_weights', action='store',
                        type=bool,
                        help='Whether to use tied weights')
    parser.add_argument('act_enc', action='store',
                        type=str,
                        choices=['tanh', 'linear', 'softplus', 'sigmoid'],
                        help='Activation function for the encoder')
    parser.add_argument('act_dec', action='store',
                        type=str,
                        choices=['tanh', 'linear', 'softplus', 'sigmoid'],
                        help='Activation function for the decoder')
    parser.add_argument('cost_type', action='store',
                        type=str,
                        choices=['CE', 'MSE'],
                        help='Cost function to use to train autoencoder')
    parser.add_argument('learning_rate', action='store',
                         type=float,
                         help='Learning rate to use (float)')
    parser.add_argument('batch_size', action='store',
                        type=int,
                        help='Minibatch size to use (integer)')
    parser.add_argument('epochs', action='store',
                         type=int,
                         help='Number of epochs (full passes through data)')
    parser.add_argument('noise_type', action='store',
                         type=str,
                         choices=['binomial', 'gaussian'],
                         help='Noise type to use for corruption')
    parser.add_argument('corruption_level', action='store',
                        type=float,
                        help='Corruption (noise) level (float)')
    parser.add_argument('-p', '--do-pca', action='store_const',
                        default=False,
                        const=True,
                        required=False,
                        help='Transform learned representation with PCA')
    parser.add_argument('-k', '--num-components', action = 'store',
                        type = int,
                        default = numpy.inf,
                        required = False,
                        help = "Only the 'n' most important PCA components" \
                            " will be preserved")
    parser.add_argument('-v', '--min-variance', action = 'store',
                        type = float,
                        default = .0,
                        required = False,
                        help = "PCA components with variance below this" \
                            " threshold will be discarded")
    # Note that hyphens ('-') in argument names are turned into underscores
    # ('_') after parsing
    parser.add_argument('-N', '--normalize-on-the-fly', action='store_const',
                        default=True,
                        const=False,
                        required=False,
                        help='Normalize on the fly')
    parser.add_argument('-s', '--save-dir', action='store',
                        type=str,
                        default='.',
                        required=False,
                        help='Directory to which to save output')
    parser.add_argument('-c', '--create-submission', action='store_const',
                        default=False,
                        const=True,
                        required=False,
                        help='Create a UTLC submission from the learned model')
    parser.add_argument('-d', '--submission-dir', action='store',
                        type=str,
                        default=None,
                        required=False,
                        help='Directory to which to save submission [defaults' \
                            ' to model output directory]')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir) or not os.path.isdir(args.save_dir):
        raise IOError('%s doesn\'t exist or is not accessible' % os.save_dir)

    main_train(args.dataset, args.save_dir, args.n_hidden,
               args.tied_weights, args.act_enc, args.act_dec,
               args.learning_rate, args.batch_size, args.epochs,
               args.cost_type, args.noise_type, args.corruption_level,
               args.normalize_on_the_fly, args.do_pca, args.num_components,
               args.min_variance, args.create_submission, args.submission_dir)
