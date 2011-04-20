import numpy as N
from theano import function, scan, shared
import theano.tensor as T
import copy
from theano.printing import Print
from theano.tensor.shared_randomstreams import RandomStreams
import theano
floatX = theano.config.floatX
import cPickle

class BR_ReconsSRBM:
    def reset_rng(self):

        self.rng = N.random.RandomState([12.,9.,2.])
        self.theano_rng = RandomStreams(self.rng.randint(2**30))
        if self.initialized:
            self.redo_theano()
    #

    def __getstate__(self):
        d = copy.copy(self.__dict__)

        #remove everything set up by redo_theano

        for name in self.names_to_del:
            if name in d:
                del d[name]

	print "WARNING: not pickling random number generator!!!!"
	del d['theano_rng']

        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        #self.redo_theano()      # todo: make some way of not running this, so it's possible to just open something up and look at its weights fast without recompiling it

    def weights_format(self):
        return 'col_major'

    def get_dimensionality(self):
        return 0

    def important_error(self):
        return 2

    def __init__(self, nvis, nhid,
                learning_rate, irange,
                init_bias_hid, mean_field_iters,
                damping_factor,
                persistent_chains, beta, gibbs_iters,
                enc_weight_decay):
        self.initialized = False
        self.reset_rng()
        self.nhid = nhid
        self.nvis = nvis
        self.learning_rate = learning_rate
        self.ERROR_RECORD_MODE_MONITORING = 0
        self.init_weight_mag = irange
        self.force_batch_size = 0
        self.init_bias_hid = init_bias_hid
        self.persistent_chains = persistent_chains
        self.mean_field_iters = mean_field_iters
        self.beta = N.cast[floatX] (beta)
        self.gibbs_iters = gibbs_iters
        self.damping_factor = damping_factor
        self.enc_weight_decay = N.cast[floatX](enc_weight_decay)
	self.names_to_del = []
        self.redo_everything()

    def set_error_record_mode(self, mode):
        self.error_record_mode = mode

    def set_size_from_dataset(self, dataset):
        self.nvis = dataset.get_output_dim()
        self.redo_everything()
        self.vis_mean.set_value( dataset.get_marginals(), borrow=False)
    #

    def get_input_dim(self):
        return self.nvis

    def get_output_dim(self):
        return self.nhid

    def redo_everything(self):
        self.initialized = True

        self.error_record = []
        self.examples_seen = 0
        self.batches_seen = 0

        self.W = shared( N.cast[floatX](self.rng.uniform(-self.init_weight_mag, self.init_weight_mag, (self.nvis, self.nhid ) ) ))
        self.W.name = 'W'

        self.c = shared( N.cast[floatX](N.zeros(self.nhid) + self.init_bias_hid) )
        self.c.name = 'c'

        self.vis_mean = shared( N.cast[floatX](N.zeros(self.nvis)))
        self.vis_mean.name = 'vis_mean'

        self.chains = shared ( N.cast[floatX]( N.zeros((self.persistent_chains,self.nvis))) )
        self.chains.name = 'chains'

        self.params = [ self.W, self.c, self.vis_mean ]

        self.redo_theano()
    #


    def expected_energy(self, V, Q):

	def f(v,q,w):
		return self.beta * (
					0.5 * T.dot(v,v) 
					- T.dot(self.vis_mean,v)
					-T.dot(v,T.dot(self.W,q))
					+0.5*T.dot(T.dot(self.W,q).T,T.dot(self.W,q))
					+0.5 * T.dot(w,q-T.sqr(q))
				    ) - T.dot(self.c,q)

        rval, updates = scan( f, sequences = [V,Q], non_sequences = [self.W_norms] )

	assert len(updates.keys()) == 0

        return T.mean(rval)

    def redo_theano(self):

	init_names = dir(self)

        self.W_T = self.W.T
        self.W_T.name = 'W.T'

        self.wprod = T.dot(self.W_T,self.W)
        self.wprod.name = 'W.T W'
        self.W_norms = T.sum(T.sqr(self.W),axis=0)
        self.W_norms.name = 'W_norms'
        self.mask = (1.0 - T.eye(self.nhid) )
        self.mask.name = 'mask'

        alpha = T.scalar()

        X = T.matrix()
        X.name = 'X'

        pos_Q = self.infer_Q(X)
        pos_Q.name = 'pos_Q'

        samples = [ self.chains ]

        for i in xrange(self.gibbs_iters):
            samples.append(self.gibbs_step(samples[-1]))

        final_sample = samples[-1]
        final_sample.name = 'final_sample'

        neg_Q = self.infer_Q(final_sample)
        neg_Q.name = 'neg_Q'

        self.run_sampling = function([X],[pos_Q, neg_Q], updates = [(self.chains,final_sample)] , name='run_sampling')

        outside_pos_Q = T.matrix()
        outside_pos_Q.name = 'outside_pos_Q'
        outside_neg_Q = T.matrix()
        outside_neg_Q.name = 'outside_neg_Q'

        obj = self.expected_energy(X,outside_pos_Q) \
            - self.expected_energy(self.chains,outside_neg_Q) \
            + self.enc_weight_decay * T.mean(T.sqr(self.W))


        grads = [ T.grad(obj,param) for param in self.params ]

        self.learn_from_samples = function([X, outside_pos_Q, outside_neg_Q, alpha], updates =
                [ (param, param - alpha * grad) for (param,grad)
                    in zip(self.params, grads) ] , name='learn_from_samples')

        self.recons_func = function([X], self.gibbs_step_exp(X) , name = 'recons_func')

	final_names = dir(self)

	self.names_to_del = [ name for name in final_names if name not in init_names ]

    def learn(self, dataset, batch_size):
	w = self.W.get_value(borrow=True)
	print 'weights summary: '+str( (w.min(),w.mean(),w.max()))
        self.learn_mini_batch(dataset.get_batch_design(batch_size))


    def error_func(self, x):
        return N.square( x - self.recons_func(x)).mean()

    def record_monitoring_error(self, dataset, batch_size, batches):
        assert self.error_record_mode == self.ERROR_RECORD_MODE_MONITORING

        errors = []

        for i in xrange(batches):
            x = dataset.get_batch_design(batch_size)
            error = self.error_func(x)
            errors.append( error )
        #


        self.error_record.append( (self.examples_seen, self.batches_seen, N.asarray(errors).mean() ) )
    #

    def reconstruct(self, x, use_noise):
        assert x.shape[0] == 1

        print 'x summary: '+str((x.min(),x.mean(),x.max()))

        #this method is mostly a hack to make the formatting work the same as denoising autoencoder
        self.truth_shared = shared(x.copy())

        if use_noise:
            self.vis_shared = shared(x.copy() + 0.15 *  self.rng.randn(*x.shape))
        else:
            self.vis_shared = shared(x.copy())

        self.reconstruction = self.recons_func(self.vis_shared.get_value())

        print 'recons summary: '+str((self.reconstruction.min(),self.reconstruction.mean(),self.reconstruction.max()))


    def gibbs_step_exp(self, V):
        base_name = V.name

        if base_name is None:
            base_name = 'anon'

        Q = self.infer_Q(V)
        H = self.sample_hid(Q)

        H.name =  base_name + '->hid_sample'

        sample =  self.vis_mean + T.dot(H,self.W_T)

        sample.name = base_name + '->sample_expectation'

        return sample


    def gibbs_step(self, V):
        base_name = V.name

        if base_name is None:
            base_name = 'anon'

        m = self.gibbs_step_exp(V)
        sample = self.theano_rng.normal(size = V.shape, avg = m,
                                    std = N.sqrt(1./self.beta), dtype = V.dtype)

        sample.name = base_name + '->sample'

        return sample

    def sample_hid(self, Q):
        return self.theano_rng.binomial(size = Q.shape, n = 1, p = Q,
                                dtype = Q.dtype)


    def infer_Q(self, V):
        base_name = V.name

        if base_name is None:
            base_name = 'anon'

        Q =  [ self.init_mean_field_step(V) ]

        for i in xrange(self.mean_field_iters - 1):
            Q.append ( self.damped_mean_field_step(V,Q[-1]) )

        for i in xrange(len(Q)):
            Q[i].name = base_name + '->Q ('+str(i)+')'

        return Q[-1]

    def init_mean_field_step(self, V):
        return T.nnet.sigmoid(self.c+self.beta*
                    (T.dot(V-self.vis_mean,self.W)
                          ))

    def damped_mean_field_step(self, V, P):

	def f(p,w):
		return - T.dot(self.W.T,T.dot(self.W,p))+w*(p-.5)

        interaction_term, updates = \
            scan( f, sequences  = P, non_sequences= self.W_norms)


	assert len(updates.keys()) == 0

        Q = T.nnet.sigmoid(self.c+self.beta *
                             (T.dot(V-self.vis_mean, self.W)
                               + interaction_term
                             )
                           )


        return self.damping_factor * P + (1.0 - self.damping_factor) * Q


    def learn_mini_batch(self, x):

        pos_Q, neg_Q = self.run_sampling(x)

        self.learn_from_samples(x, pos_Q, neg_Q, self.learning_rate)

        self.examples_seen += x.shape[0]
        self.batches_seen += 1
    #
#

