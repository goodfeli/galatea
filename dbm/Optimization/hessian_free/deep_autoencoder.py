"""
Copyright (c) 2011, Yahoo! Inc.  All rights reserved.
Copyright (c) 2008--2010, Theano Development Team, All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the following
conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of Yahoo! Inc. nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of Yahoo! Inc.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression, load_data
from mlp_tutorial import HiddenLayer
from rbm import RBM

from hf import truncated_newton
import numpy

# Use the MRG random number generator when on the GPU
if 'gpu' in theano.config.device:
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
else:
    from theano.tensor.shared_randomstreams import RandomStreams

def build_score_fn(inputs,index,costs,xdata,ydata,batch_size,updates,obj):
    """
    a general purpose helper function to build theano function:
    
    """
    return theano.function(inputs = inputs,
                           outputs = costs,
                           updates = updates,
                           givens  = {
                               obj.x : xdata[index*batch_size:(index+1)*batch_size],
                               obj.y : ydata[index*batch_size:(index+1)*batch_size]
                           },
                           on_unused_input='warn')

class DeepAutoencoder(object):

    def __init__(self, numpy_rng, theano_rng = None, n_ins = 784,
                 hidden_layers_sizes = [50,60], n_outs = 10,
                 reconstruction_cost = 'cross_entropy',\
                 supervised_training = 'russ',
                 tied_weights = False):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []
        self.reconstruction_layers = []
        self.rbm_layers     = []
        self.params         = []
        self.n_layers       = len(hidden_layers_sizes)
        self.tied_weights   = tied_weights

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))

        # allocate symbolic variables for the data
        self.x  = T.matrix('x')  # the data is presented as rasterized images
        self.y  = T.ivector('y') # the labels are presented as 1D vector of
                                 # [int] labels

        # The DBN is an MLP, for which all weights of intermediate layers are shared with a
        # different RBM.  We will first construct the DBN as a deep multilayer perceptron, and
        # when constructing each sigmoidal layer we also construct an RBM that shares weights
        # with that layer. During pretraining we will train these RBMs (which will lead
        # to chainging the weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the MLP.

        for i in xrange( self.n_layers ):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of the layer below or
            # the input size if we are on the first layer
            if i == 0 :
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i-1]

            # the input to this layer is either the activation of the hidden layer below or the
            # input of the DBN if you are on the first layer
            if i == 0 :
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            # TODO(dumitru): this is temporary, get rid of this when done
            if i != self.n_layers-1:
               activation = T.nnet.sigmoid
            else:
               activation = None

            sigmoid_layer = HiddenLayer(rng   = numpy_rng,
                                           input = layer_input,
                                           n_in  = input_size,
                                           n_out = hidden_layers_sizes[i],
                                           activation = activation)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # its arguably a philosophical question...  but we are going to only declare that
            # the parameters of the sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not of the DBN.
            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng = numpy_rng, theano_rng = theano_rng,
                          input = layer_input,
                          n_visible = input_size,
                          n_hidden  = hidden_layers_sizes[i],
                          W = sigmoid_layer.W,
                          hbias = sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # Creating the reconstruction layers
        for i in xrange(self.n_layers,0,-1):

            if i == self.n_layers:
                layer_input = self.sigmoid_layers[-1].output
            else:
                layer_input = self.reconstruction_layers[-1].output

            if self.tied_weights:
               W = self.sigmoid_layers[i-1].W.T
               b = self.rbm_layers[i-1].vbias
            else:
               W = b = None

            # the output size if we are on the first layer
            if i == 1 :
                output_size = n_ins
                activation = None
            else:
                output_size = hidden_layers_sizes[i-2]
                activation = T.nnet.sigmoid

            input_size = hidden_layers_sizes[i-1]

            reconstruction_layer = HiddenLayer(numpy_rng, layer_input, input_size, output_size, W, b, activation)

            if not self.tied_weights:

               self.params.extend(reconstruction_layer.params)

            # add the layer to our list of layers
            self.reconstruction_layers.append(reconstruction_layer)

        self.xhat = T.nnet.sigmoid( self.reconstruction_layers[-1].output )

        # otherwise we'll end up with a bunch of extra params and theano will complain
        self.reconstruction_params = list(self.params)

        self.global_cross_entropy =  T.mean(-T.sum( self.x*T.log(self.xhat) + (1-self.x)*T.log(1-self.xhat), axis=1 ))
        self.global_mse = T.mean(T.sum((self.x - self.xhat)**2.0,axis=1))

        if reconstruction_cost == 'cross_entropy':
           self.global_pretraining_cost = self.global_cross_entropy
        elif reconstruction_cost == 'mse':
            self.global_pretraining_cost = self.global_mse
        else:
            raise NotImplementedError('Invalid reconstruction error\
                                      specified')


        if supervised_training == 'standard' or supervised_training == 'hf':
           # We now need to add a logistic layer on top of the MLP
           self.logLayer = LogisticRegression(\
                         input = self.sigmoid_layers[-1].output,\
                         n_in = hidden_layers_sizes[-1], n_out = n_outs)
           self.params.extend(self.logLayer.params)

           # compute the cost for second phase of training, defined as the
           # negative log likelihood of the logistic regression (output) layer
           self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

           # compute the gradients with respect to the model parameters
           # symbolic variable that points to the number of errors made on the
           # minibatch given by self.x and self.y
           self.errors = self.logLayer.errors(self.y)

           # compute prediction as class whose probability is maximal in
           # symbolic form
           #self.y_pred=T.argmax(self.p_y_given_x, axis=1)


        elif supervised_training == 'russ':

           # compute vector of class-membership probabilities in symbolic form
           self.p_y_given_x = T.nnet.softmax(self.sigmoid_layers[-1].output)
           self.y_pred=T.argmax(self.p_y_given_x, axis=1)
           self.finetune_cost = -T.mean(T.log(self.p_y_given_x[T.arange(self.y.shape[0]),self.y]))
           self.errors = T.mean(T.neq(self.y_pred, self.y))

        else:
           print 'Unsupport supervised training method', supervised_training
           raise NotImplementedError

    def build_pretraining_functions(self, train_set_x, batch_size,k):
        ''' Generates a list of functions, one function per layer,
        for performing one step of gradient descent at a
        given layer. The function will require as input the minibatch index, and to train an
        RBM you just need to iterate, calling the corresponding function on all minibatch
        indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        return
        ------
        a list of pretraining functions for each layer
        '''

        # index to a [mini]batch
        index            = T.lscalar('index')   # index to a minibatch
        learning_rate    = T.scalar('lr')    # learning rate to use
        weight_decay     = T.scalar('weight_decay')    # learning rate to use


        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin+batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            
            cost,updates = rbm.get_cost_updates(learning_rate, persistent=None, k = k, weight_decay = weight_decay)

            # compile the theano function
            fn = theano.function(inputs = [index,
                              theano.Param(learning_rate, default = 0.1),
                              theano.Param(weight_decay, default = 0.0002)],
                    outputs = cost,
                    updates = updates,
                    givens  = {self.x :train_set_x[batch_begin:batch_end]},
                    on_unused_input='warn')
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_global_pretraining_functions_sgd(self, datasets, train_batch_size, learning_rate):
        """
        to minimize the reconstruction error

        return:
        -
        -train_fn, for just one minibatch
        -valid_fn, for all minibatches
        -test_fn, for all minibatches 
        """
        
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x , test_set_y ) = datasets[2]

        # compute number of minibatches5B for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / train_batch_size
        n_test_batches  = test_set_x.get_value(borrow=True).shape[0]  / train_batch_size

        index   = T.lscalar('index')    # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.global_pretraining_cost, self.reconstruction_params)

        # compute list of updates
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam*learning_rate

        costs = [self.global_pretraining_cost,self.global_mse]

        train_fn = build_score_fn([index],index,costs,train_set_x,train_set_y,train_batch_size,updates,self)

        # test and valid do not contain updates
        test_score_i = build_score_fn([index],index,costs,test_set_x,test_set_y,train_batch_size,[],self)
        valid_score_i = build_score_fn([index],index,costs,valid_set_x,valid_set_y,train_batch_size, [],self)

        # Create a function that scans the entire validation set
        def valid_fn():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_fn():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_fn, test_fn


    def build_finetune_functions(self, datasets, train_batch_size, learning_rate):
        '''Generates a function `train` that implements one step of finetuning, a function
        `validate` that computes the error on a batch from the validation set, and a function
        `test` that computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;  the has to contain three
        pairs, `train`, `valid`, `test` in this order, where each pair is formed of two Theano
        variables, one for the datapoints, the other for the labels
        :type train_batch_size: int
        :param train_batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        return
        ------
        train_fn, valid_fn, test_fn
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x , test_set_y ) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / train_batch_size
        n_test_batches  = test_set_x.get_value(borrow=True).shape[0]  / train_batch_size

        index   = T.lscalar('index')    # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam*learning_rate

        costs = self.finetune_cost

        train_fn = build_score_fn([index],index,costs,train_set_x,train_set_y,train_batch_size,updates,self)
        test_score_i = build_score_fn([index],index,costs,test_set_x,test_set_y,train_batch_size,[],self)
        valid_score_i = build_score_fn([index],index,costs,valid_set_x,valid_set_y,train_batch_size,[],self)

        # Create a function that scans the entire validation set
        def valid_fn():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_fn():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_fn, test_fn

    def build_finetune_functions_hf(self, datasets, train_batch_size,
                                    preconditioner, ridge, maxiterations):
        
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x , test_set_y ) = datasets[2]

        # compute number of minibatches for training, validation and testing
        index   = T.lscalar('index')    # index to a [mini]batch

        valid_batch_size = valid_set_x.shape[0]
        test_batch_size = test_set_x.shape[0]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / valid_batch_size
        n_test_batches  = test_set_x.get_value(borrow=True).shape[0]  / test_batch_size
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / train_batch_size

        costs = [self.finetune_cost, self.errors]
        # 
        def train_fn():
            
            givens = {
                self.x:train_set_x[index*train_batch_size:(index+1)*train_batch_size],
                self.y:train_set_y[index*train_batch_size:(index+1)*train_batch_size]}

            error = truncated_newton([index], self.logLayer.p_y_given_x,
                                     costs, self.params, givens,
                                     maxiterations, ridge, preconditioner, n_train_batches)
            return error

        test_score_i = build_score_fn([index],index,costs,test_set_x,test_set_y,valid_batch_size,[],self)
        valid_score_i = build_score_fn([index],index,costs,valid_set_x,valid_set_y,test_batch_size,[],self)

        def valid_fn():
            
            #return [valid_score_i(i) for i in range(n_valid_batches)]
            return [valid_score_i(i) for i in range(1)]

        def test_fn():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_fn, test_fn

    def build_global_pretraining_functions_hf(self, datasets, train_batch_size,
                                    preconditioner, ridge, maxiterations):
        """
        params:
        -------
        datasets: train, valid, test
        train_batch_size:
        preconditioner:
        ridge: lamda(damping parameter ofr H) in the paper
        maxiterations:
        
        return
        ------
        train_fn, valid_fn, test_fn
        """
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x , test_set_y ) = datasets[2]

        # compute number of minibatches for training, validation and testing
        index   = T.lscalar('index')    # index to a [mini]batch

        valid_batch_size = valid_set_x.shape[0]
        test_batch_size = test_set_x.shape[0]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / valid_batch_size
        n_test_batches  = test_set_x.get_value(borrow=True).shape[0]  / test_batch_size
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / train_batch_size

        costs = [self.global_pretraining_cost,self.global_mse]

        # 
        def train_fn():
            
            givens = {
                self.x:train_set_x[index*train_batch_size:(index+1)*train_batch_size],
                self.y:train_set_y[index*train_batch_size:(index+1)*train_batch_size]}

            error = truncated_newton([index], self.reconstruction_layers[-1].output,
                                     costs, self.reconstruction_params, givens,
                                     maxiterations, ridge, preconditioner, n_train_batches)
            return error


        test_score_i = build_score_fn([index],index,costs,test_set_x,test_set_y,valid_batch_size,[],self)
        valid_score_i = build_score_fn([index],index,costs,valid_set_x,valid_set_y,test_batch_size,[],self)

        def valid_fn():
            
            #return [valid_score_i(i) for i in range(n_valid_batches)]
            return [valid_score_i(i) for i in range(1)]

        def test_fn():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_fn, test_fn

    def initialize_reconstruction_weights(self):
        if not self.tied_weights:
            updates = []
            for i in xrange(self.n_layers,0,-1):
               updates.append((self.reconstruction_layers[self.n_layers-i].W, self.sigmoid_layers[i-1].W.T))
               updates.append((self.reconstruction_layers[self.n_layers-i].b, self.rbm_layers[i-1].vbias))

            f = theano.function([],[],updates = updates, on_unused_input='warn')
            
    def save_reconstruction_weights(self, filename):
        if not self.tied_weights:
            param_list = []
            for i in xrange(self.n_layers, 0, -1):
                params = (self.reconstruction_layers[self.n_layers-i].W.get_value(),
                          self.reconstruction_layers[self.n_layers-i].b.get_value())
                param_list.append(params)
            import cPickle
            cPickle.dump(param_list, open(filename, 'w'), -1)
            
    def load_reconstruction_weights(self, filename):
        import cPickle
        param_list = cPickle.load(open(filename, 'r'))
        for i in range(self.n_layers):
            W, b = param_list[i]
            self.reconstruction_layers[self.n_layers-i].W.set_value(numpy.array(W, dtype=theano.config.floatX))
            self.reconstruction_layers[self.n_layers-i].b.set_value(numpy.array(b, dtype=theano.config.floatX))
                                                                    
    def save_rbm_weights(self, filename):
        param_list = []
        for i in range(self.n_layers):
           params = (self.rbm_layers[i].W.get_value(), self.rbm_layers[i].vbias.get_value(), self.rbm_layers[i].hbias.get_value())
           param_list.append(params)
        import cPickle
        cPickle.dump(param_list,open(filename,'w'), -1)

    def load_rbm_weights(self, filename):
        import cPickle
        param_list = cPickle.load(open(filename,'r'))
        for i in range(self.n_layers):
           W,vbias,hbias = param_list[i]
           self.rbm_layers[i].W.set_value(numpy.array(W,dtype=theano.config.floatX))
           self.rbm_layers[i].vbias.set_value(numpy.array(vbias,dtype=theano.config.floatX))
           self.rbm_layers[i].hbias.set_value(numpy.array(hbias,dtype=theano.config.floatX))

