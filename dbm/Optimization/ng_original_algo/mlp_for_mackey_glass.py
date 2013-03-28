import numpy
import theano
import theano.tensor as T

floatX = theano.config.floatX

class MLP_Mackey_Glass:
    def __init__(self, rng, input, n_in, n_hidden, n_out,
                 hidden_activation='sigmoid',
                 output_activation='linear', ):
        
        # init value for hidden layer
        W_h_init = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_hidden)),
                    high=numpy.sqrt(6. / (n_in + n_hidden)),
                    size=(n_in, n_hidden)), dtype=floatX)
        if hidden_activation == 'sigmoid':
            W_h_init *= 4
        b_h_init = numpy.zeros((n_hidden,), dtype=floatX)
        
        # init value for output layer
        W_o_init = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_hidden + n_out)),
                    high=numpy.sqrt(6. / (n_hidden + n_out)),
                    size=(n_hidden, n_out)), dtype=floatX)
        
        b_o_init = numpy.zeros((n_out,), dtype=floatX)

        init_values = [W_h_init, b_h_init, W_o_init, b_o_init]
        init_values = [param.flatten() for param in init_values]
        
        # create theta that store all params as a vector
        # W_h, b_h, W_o, b_o
        length = [n_in*n_hidden, n_hidden, n_hidden*n_out, n_out]
        self.index = [0]
        for i in length:
            self.index.append(i + self.index[-1])

        inits = numpy.concatenate([init_values[0], init_values[1],
                                   init_values[2], init_values[3]])
        
        self.theta = theano.shared(numpy.asarray(inits, dtype=floatX))
        
        self.W_h = self.theta[self.index[0]:self.index[1]].reshape((n_in, n_hidden))
        self.b_h = self.theta[self.index[1]:self.index[2]].reshape((n_hidden,))
        self.W_o = self.theta[self.index[2]:self.index[3]].reshape((n_hidden, n_out))
        self.b_0 = self.theta[self.index[3]:self.index[4]].reshape((n_out,))
        # get hidden outputs
        self.outputs_linear_h = T.dot(input, self.W_h) + self.b_h 
        if hidden_activation == 'sigmoid':
            self.outputs_h = T.nnet.sigmoid(self.outputs_linear_h)
        else:
            self.outputs_h = T.tanh(self.outputs_linear_h)

        # get predict
        self.predict = T.dot(self.outputs_h, self.W_o) + self.b_0
        