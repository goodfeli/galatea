import numpy, time, cPickle, gzip, sys, os, copy

import theano
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from dense.logistic_sgd import load_data
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
                 tied_weigths = True,
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
        self.tied_weights = tied_weigths

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
            return  self.theano_rng.binomial( size = input.shape, n = 1, p =  1 - corruption_level, dtype=theano.config.floatX) * input
        elif noise == 'gaussian':
            return input + self.theano_rng.normal( size = input.shape, avg=0, std = corruption_level, dtype=theano.config.floatX)
        else:
            raise NotImplementedError('This noise %s is not implemented yet'%(noise))
    
    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        if self.act_enc == 'sigmoid':
            return T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        elif self.act_enc == 'tanh':
            return T.tanh(T.dot(input, self.W) + self.b)
        else:
            raise NotImplementedError('Encoder function %s is not implemented yet'%(self.act_enc))

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
            raise NotImplementedError('Decoder function %s is not implemented yet'%(self.act_dec))

    def get_cost_updates(self, corruption_level, learning_rate, cost = 'CE', noise = 'binomial'):
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
            raise NotImplementedError('This cost function %s is not implemented yet'%(cost))

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
            noise='gaussian', corruption_level=0.3, normalize=True):
        """ This function fits the dA to the dataset given
        some hyper-parameters and returns the loss evolution
        and the time spent during training   """

	
        # compute number of minibatches for training, validation and testing
        n_train_batches = dataset.value.shape[0] / batch_size
	
        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch 

        cost, updates = self.get_cost_updates(corruption_level = corruption_level,
                                learning_rate = learning_rate,
                                noise = noise,
                                cost = cost)
        if normalize:
            train_da = theano.function([index], cost, updates = updates,
                givens = {self.x:dataset[index*batch_size:(index+1)*batch_size]})
    	else:
            if dataset.value.shape[1]==7200:
                #q&d pour detecter rita
                max=float(dataset.value.max())
            else:
                max=0.69336046033925791
                #0.69336046033925791 std for harry
            datasetB = theano.shared(numpy.asarray(dataset.value[0:batch_size], dtype=theano.config.floatX))
            train_da = theano.function([], cost, updates = updates,
                    givens = {self.x:datasetB})

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
            for batch_index in xrange(n_train_batches):
                if normalize:
        	        c.append(train_da(batch_index))
                else:
                    datasetB.value = dataset.value[batch_index*batch_size:(batch_index+1)*batch_size]/max
                    c.append(train_da())
            toc = time.clock()
            loss.append(numpy.mean(c))
            print 'Training epoch %d, time spent (min) %f,  cost '%(epoch,(toc-tic)/60.), numpy.mean(c)
            toc = tic

        end_time = time.clock()
        training_time = (end_time - start_time)/60.

        return training_time, loss

    def save(self, save_dir):
        """ save the parameters of the model """
        print '... saving model'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        save_file = open(save_dir + 'model.pkl','wb')
        cPickle.dump(self.__initargs__, save_file, -1)
        cPickle.dump(self.W.value, save_file, -1)
        if not self.tied_weights:
            cPickle.dump(self.W_prime.value, save_file, -1)
        cPickle.dump(self.b.value, save_file, -1)
        cPickle.dump(self.b_prime.value, save_file, -1)
        save_file.close()
 
    def load(self, load_dir):
        """ load the model """
        print '... loading model'
        save_file = open(load_dir + 'model.pkl','r')
        args = cPickle.load(save_file)
        self.__init__(
                 seed_params = args['seed_params'],
                 seed_noise = args['seed_noise'],
                 input = args['input'],
                 n_visible= args['n_visible'],
                 n_hidden= args['n_hidden'], 
                 tied_weigths = args['tied_weigths'],
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
        n_train_batches = dataset.value.shape[0] / batch_size

        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch 

        cost, updates = self.get_cost_updates(corruption_level = corruption_level,
                                learning_rate = 0.,
                                noise = noise,
                                cost = cost)


        get_error = theano.function([index], cost, updates = {},
            givens = {self.x:dataset[index*batch_size:(index+1)*batch_size]})

        denoising_error = []
        # go through the dataset
        for batch_index in xrange(n_train_batches):
            denoising_error.append(get_error(batch_index))

        return numpy.mean(denoising_error)

      

def main_train(dataset, save_dir, n_hidden, tied_weights, act_enc,
    act_dec, learning_rate, batch_size, epochs, cost_type,
    noise_type, corruption_level,normalize=True):
    ''' main function used for training '''

    datasets = load_data(dataset,normalize)
    train_set_x = datasets[0]

    d = train_set_x.value.shape[1]
    
    da = dA(n_visible = d, n_hidden = n_hidden, 
            tied_weigths = tied_weights,
            act_enc = act_enc, act_dec = act_dec)

    time_spent, loss = da.fit(train_set_x, learning_rate, batch_size, epochs, cost_type,
            noise_type, corruption_level,normalize)

    if save_dir:
        da.save(save_dir)

    denoising_error = da.get_denoising_error(train_set_x, cost_type,
        noise_type, corruption_level)
    print 'Training complete in %f (min) with final denoising error %f'%(time_spent,denoising_error)
    return denoising_error, time_spent, loss

if __name__ == '__main__':
    # you can train a denoising autoencoder using this cmd:
    # python <thisfile> dataset #hidden_units tied_weights act_enc act_dec
    # costtype learning_rate batchsize epochs noise_type corruption_level
    #
    # here a few examples
    #
    # python dA.py avicenna 500 True 'sigmoid' 'linear' 'MSE' 0.01 20 50 'gaussian' 0.3
    # attention ajouter un zero a la fin de la commande pour rita qui indique que
    # les donnes ne sont pas normalisees et qu'il y a un hack !!
    # python dA.py rita 500 True 'sigmoid' 'sigmoid' 'CE' 0.01 20 50 'gaussian' 0.3 0 
    # python dA.py sylvester 500 True 'sigmoid' 'linear' 'MSE' 0.01 20 50 'gaussian' 0.3
    # python dA.py ule 500 True 'sigmoid' 'sigmoid' 'CE' 0.01 1 50 'gaussian' 0.3
    #
    # Pour harry si l'on n'exploite pas la spacité on peut lancer avec normalisation à 
    # la volée (0 à la fin comme rita)
    # python dA.py harry 500 True 'sigmoid' 'sigmoid' 'CE' 0.01 20 50 'gaussian' 0.3 0 
    #
    dataset = sys.argv[1]
    n_hidden = int(sys.argv[2])
    tied_weights = bool(sys.argv[3])
    act_enc= sys.argv[4]
    act_dec= sys.argv[5]
    cost_type= sys.argv[6]
    learning_rate= float(sys.argv[7])
    batch_size= int(sys.argv[8])
    epochs= int(sys.argv[9])
    noise_type= sys.argv[10]
    corruption_level = float(sys.argv[11])
    save_dir = './'

    if (len(sys.argv) > 12):   # loading un-normalized data in memory (rita)
        normalize = bool(int(sys.argv[12]))
        if not dataset in ['rita','harry']:
            raise NotImplementedError('for now the normalization on the fly is only allowed for rita & harry, may change...')

    else:
        normalize=True

    main_train(dataset, save_dir, n_hidden, tied_weights, act_enc,
        act_dec, learning_rate, batch_size, epochs, cost_type,
        noise_type, corruption_level,normalize)
