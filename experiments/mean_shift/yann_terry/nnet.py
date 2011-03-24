"""
An MLP for multiple instance classification problems.
"""
import numpy
import theano
from theano import tensor as T
from theano import sparse as S
from theano.tensor.shared_randomstreams import RandomStreams
import scipy.sparse


class NeuralNetworkLayer(object):
    """
    A Neural Network layer with that takes the activation function as parameter.
    """
    def __init__(self, x, n_in, n_out, activation, weight_sparsity=False, output_sparsity=False, sparsity_map=None, W=None, b=None, disable_l1act=False, sampling_pattern=None):
        self.n_in = n_in
        self.n_out = n_out
        self.input_sparsity = isinstance(x.type, S.SparseType)
        self.weight_sparsity = weight_sparsity or (W != None and isinstance(W.value, scipy.sparse.spmatrix))
        self.output_sparsity = output_sparsity
        self.sparsity_map = sparsity_map
        self.disable_l1act = disable_l1act
        
        if W == None:
            W_values = numpy.asarray( numpy.random.uniform(
                low  = - numpy.sqrt(6./(n_in+n_out)),
                high =   numpy.sqrt(6./(n_in+n_out)),
                size = (n_in, n_out) if sampling_pattern == None else (n_out, n_in)), dtype = theano.config.floatX)
        
            if weight_sparsity:
                W_values *= self.get_sparsity_map(W_values.shape, weight_sparsity)
                self.W = theano.shared(scipy.sparse.csr_matrix(W_values), name = 'W%dx%d' % (n_in, n_out), format="csr")
            else:
                self.W = theano.shared(W_values, name = 'W%dx%d' % (n_in, n_out))
        else:
            self.W = W
        
        if b == None:
            self.b = theano.shared(numpy.zeros((n_out,), dtype=theano.config.floatX), name = 'b')
        else:
            self.b = b
        
        if sampling_pattern != None:
            self.output = activation(S.sampling_dot(x, self.W, sampling_pattern) + self.b)
        elif self.weight_sparsity or self.input_sparsity:
          self.output = activation(S.dot(x, self.W) + self.b)
        else:
          self.output = activation(T.dot(x, self.W) + self.b)
        
        if output_sparsity:
            self.output = S.csr_from_dense(self.output)
    
    
    def get_gaussian_sparsity_map(self, shape, sparsity, preference_sigma=0.1, connections_sigma=0.1):
        preferences = numpy.random.normal(1, preference_sigma, shape[0])
        connections = (1 - sparsity) * shape[0] * numpy.random.normal(1, connections_sigma, shape[1])
        
        w = map(lambda n: map(lambda p: numpy.random.binomial(1, p), n * preferences / shape[0]), connections)
        
        return numpy.array(w, dtype=theano.config.floatX).T
    
    
    def get_uniform_sparsity_map(self, shape, sparsity):
        return numpy.asarray(numpy.random.binomial(1, (1 - sparsity), size=shape), dtype=theano.config.floatX)
    

    def get_tiled_sparsity_map(self, shape, sparsity):
        def get_unit(size, sparsity_):
            res = numpy.zeros(size)

            # Allow for little jitter in sparsity
            # sparsity_ = max(min(numpy.random.normal(sparsity, 0.02), 1.), 0.)

            nonzero_w = numpy.random.randint(int(numpy.sqrt(1 - sparsity_) * size[1] / 1.5), int(1.5 * numpy.sqrt(1 - sparsity_) * size[1]))
            nonzero_h = nonzero_h = int((1 - sparsity_) * size[0] * size[1] / nonzero_w)

            nonzero_i = numpy.random.randint(0, size[1] - nonzero_w) if nonzero_w != size[1] else 0
            nonzero_j = numpy.random.randint(0, size[0] - nonzero_h) if nonzero_h != size[0] else 0

            res[nonzero_j:(nonzero_j+nonzero_h), nonzero_i:(nonzero_i+nonzero_w)] = 1.

            return res
        
        if int(numpy.sqrt(shape[0]))**2 != shape[0]:
          # TODO: Support more types.
          raise "Only square input images supported!!"

        im_size = int(numpy.sqrt(shape[0])) 

        w = numpy.concatenate([[get_unit((im_size, im_size), sparsity).flatten()] for _ in range(shape[1])], axis=0).T
        w = numpy.asarray(w, dtype=theano.config.floatX)

        return w

    def get_sparsity_map(self, shape, sparsity):
        mapping = {
            'G' : self.get_gaussian_sparsity_map,
            'U' : self.get_uniform_sparsity_map,
            'T' : self.get_tiled_sparsity_map,
            None : self.get_uniform_sparsity_map,
        }

        return mapping[self.sparsity_map](shape, sparsity)
    
    
    def reset(self):
        """
        Reset the weights to suitable random values.
        """
        W_values = numpy.asarray( numpy.random.uniform(
            low  = - numpy.sqrt(6./(self.n_in+self.n_out)),
            high =   numpy.sqrt(6./(self.n_in+self.n_out)),
            size = (self.n_in, self.n_out)), dtype = theano.config.floatX)
        
        if self.weight_sparsity:
            W_values *= self.get_sparsity_map(W_values.shape, self.weight_sparsity)
            W_values = scipy.sparse.csr_matrix(W_values)
        
        self.b.value = numpy.zeros((self.n_out,), dtype=theano.config.floatX)
        self.W.value = W_values


class LogisticLayer(NeuralNetworkLayer):
    """
    A Neural Network layer with a logistic activation function.
    """
    def __init__(self, x, n_in, n_out, **kwargs):
        super(LogisticLayer, self).__init__(x, n_in, n_out, T.nnet.sigmoid, **kwargs)


class RectifierLayer(NeuralNetworkLayer):
    """
    A Neural Network layer with a rectifier activation function.
    """
    def __init__(self, x, n_in, n_out, mask=True, **kwargs):
        if mask:
          activation = lambda x: numpy.asarray(numpy.sign(numpy.random.uniform(low=-1, high=1, size=(n_out,))), dtype=theano.config.floatX) * T.maximum(x, 0)
        else:
          activation = lambda x: T.maximum(x, 0)
        super(RectifierLayer, self).__init__(x, n_in, n_out, activation, **kwargs)


class TanhLayer(NeuralNetworkLayer):
    """
    A Neural Network layer with a tanh activation function.
    """
    def __init__(self, x, n_in, n_out, **kwargs):
        super(TanhLayer, self).__init__(x, n_in, n_out, T.tanh, **kwargs)


class SoftmaxLayer(NeuralNetworkLayer):
    """
    A Neural Network layer with a logistic activation function.
    """
    def __init__(self, x, n_in, n_out, **kwargs):
        super(SoftmaxLayer, self).__init__(x, n_in, n_out, T.nnet.softmax, **kwargs)


class LinearLayer(NeuralNetworkLayer):
    """
    A Neural Network layer with a linear activation function.
    """
    def __init__(self, x, n_in, n_out, **kwargs):
        super(LinearLayer, self).__init__(x, n_in, n_out, lambda x: x, **kwargs)


class DoubleRectifierLayer(NeuralNetworkLayer):
    """
    A Neural Network layer with a linear activation function.
    """
    def __init__(self, x, n_in, n_out, **kwargs):
        super(DoubleRectifierLayer, self).__init__(x, n_in, n_out, lambda x: T.maximum(0, T.minimum(1, x)), **kwargs)

class NeuralNetworkTrainer(object):
    """
    Train the given layers of a neural network with the given loss function.
    """
    def __init__(self, x, loss, layers, y=None, learning_rate=0.01, l1=0, l2=0, l1act=0):
        l1_loss = sum([abs(layer.W).sum() for layer in layers if not layer.weight_sparsity])
        l2_loss = sum([(layer.W**2).sum() for layer in layers if not layer.weight_sparsity])
        l1act_loss = sum([abs(layer.output).sum() for layer in layers if not layer.disable_l1act])
        cost = loss + l1 * l1_loss + l2 * l2_loss + l1act * l1act_loss
        
        params = [layer.W for layer in layers] + [layer.b for layer in layers]
                
        gparams = []
        for param in params:
            gparam  = S.grad(cost, param)
            gparams.append(gparam)

        updates = []
        for param, gparam in zip(params, gparams):
            if param in [layer.W for layer in layers if layer.weight_sparsity]:
              updates.append((param, param - S.sp_ones_like(param) * (learning_rate * gparam)))
            else:
              updates.append((param, param - learning_rate * gparam))
        
        args = [x] if y == None else [x, y]
        
        self._train = theano.function(args, outputs=cost, updates=updates)
    
    
    def train(self, x, y=None):
        if y == None:
            return self._train(x)
        else:
            return self._train(x, y)


class MLP(object):
    def __init__(self, n_in, layers, corruption_levels=0.2, corruption_types="B", decoder_types="L", sampling=None, **kwargs):
        x = S.csr_fmatrix('x')
        y = T.lvector('y')
        
        type_map = {
            'L' : LogisticLayer,
            'R' : RectifierLayer,
            'S' : SoftmaxLayer,
            'Li' : LinearLayer,
            'D' : DoubleRectifierLayer,
        }

        self.layers = []
        
        # Create hidden layers
        for i, layer in enumerate(layers):
            layer_type = layer[0]
            layer_size = layer[1]
            layer_weight_sparsity = layer[2] if len(layer) > 2 else False
            layer_sparsity_map = layer[3] if len(layer) > 3 else None
            layer_output_sparsity = layer[4] if len(layer) > 4 else False

            if i == 0:
                layer_input = x
                layer_n_in = n_in
            else:
                layer_input = self.layers[-1].output
                layer_n_in = self.layers[-1].n_out
            
            xargs = {}
            
            if layer_type == 'R' and layer == layers[-1]:
                xargs['mask'] = False

            layer = type_map[layer_type](layer_input,
                                         layer_n_in,
                                         layer_size,
                                         weight_sparsity=layer_weight_sparsity,
                                         output_sparsity=layer_output_sparsity,
                                         sparsity_map=layer_sparsity_map,
                                         **xargs)

            self.layers.append(layer)
        
        # Output of the network
        output = self.layers[-1].output 
        
        self._output = theano.function([x], T.argmax(output, axis=1))
        
        self._outputp = theano.function([x], output)

        # Loss
        loss = -T.mean(T.log(output)[T.arange(y.shape[0]), y])
        
        # Trainer
        self.trainer = NeuralNetworkTrainer(x, loss, self.layers, y=y, **kwargs)
        
        # Pretraining
        self.pretrainers = []
        
        theano_rng = RandomStreams(numpy.random.randint(0, 2**30))
        
        for i, layer in enumerate(self.layers):
            input = x if i == 0 else self.layers[i - 1].output
            
            target = S.dense_from_sparse(input) if i == 0 else input
            
            decoder_type = decoder_types if type(decoder_types) is str else decoder_types[i]
            
            corruption_level = corruption_levels if type(corruption_levels) is float else corruption_levels[i]
            
            corruption_type = corruption_types if type(corruption_types) is str else corruption_types[i]

            if corruption_type == "B":
                noisy_input = theano_rng.binomial(size=target.shape, n=1, p=(1 - corruption_level), dtype=theano.config.floatX) * input
            else:
                noisy_input = theano_rng.normal(size=target.shape, avg=0.0, std=corruption_level, dtype=theano.config.floatX) + input
            
            if sampling != None and i == 0:
                p = T.cast((input + theano_rng.binomial(size=target.shape, n=1, p=sampling, dtype=theano.config.floatX)) > 0, dtype=theano.config.floatX)
            else:
                p = None

            encoder = layer.__class__(noisy_input, layer.n_in, layer.n_out, W=layer.W, b=layer.b)
            
            decoder = type_map[decoder_type](encoder.output, layer.n_out, layer.n_in, disable_l1act=True, sampling_pattern=p)
            
            if p != None:
                pretrain_loss = T.mean(-T.sum( p * (target*T.log(decoder.output) + (1.-target)*T.log(1-decoder.output)), axis=1))
            else:
                pretrain_loss = T.mean(-T.sum((target*T.log(decoder.output) + (1.-target)*T.log(1.-decoder.output)), axis=1))
            
            self.pretrainers.append(NeuralNetworkTrainer(x, pretrain_loss, [encoder, decoder], **kwargs))
    
    
    def pretrain(self, x, layer):
        return self.pretrainers[layer].train(x)
    
    
    def train(self, x, y):
        return self.trainer.train(x, y)


    def output(self, x, batch_size=None):
        if batch_size:
            out = []
            n_batches = x.shape[0] / batch_size
            for n in range(n_batches):
                out.append(self._output(x[n * batch_size : (n+1) * batch_size ]))
            return numpy.concatenate(out)
        else:
            return self._output(x)
    
    
    def outputp(self, x):
        return self._outputp(x)
    
    
    def save(self):
        for i, layer in enumerate(self.layers):
            numpy.save("W%d.npy" % i, layer.W.value)
            numpy.save("b%d.npy" % i, layer.b.value)
    
    
    def reset(self):
        for layer in self.layers:
            layer.reset()

