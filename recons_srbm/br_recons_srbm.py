import numpy as N
from theano import function, scan, shared
import theano.tensor as T
import copy
from theano.printing import Print
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import time
from framework.instrument_record import InstrumentRecord
floatX = theano.config.floatX

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
        return ['v','h']

    def get_dimensionality(self):
        return 0

    def important_error(self):
        return 2

    def __init__(self, nvis, nhid,
                learning_rate, irange,
                init_bias_hid, mean_field_iters,
                damping_factor,
                no_damp_iters,
                persistent_chains, init_beta, learn_beta, beta_lr_scale, gibbs_iters,
                enc_weight_decay, fold_biases = False,
                use_cd = False, instrumented = False):
        self.initialized = False
        self.reset_rng()
        self.nhid = nhid
        self.nvis = nvis
        self.learning_rate = learning_rate
        self.ERROR_RECORD_MODE_MONITORING = 0
        self.error_record_mode = self.ERROR_RECORD_MODE_MONITORING
        self.init_weight_mag = irange
        self.force_batch_size = 0
        self.init_bias_hid = init_bias_hid
        self.persistent_chains = persistent_chains
        self.mean_field_iters = mean_field_iters
        self.no_damp_iters = no_damp_iters
        self.gibbs_iters = gibbs_iters
        self.damping_factor = damping_factor
        self.enc_weight_decay = N.cast[floatX](enc_weight_decay)
        self.names_to_del = []
        self.fold_biases = fold_biases
        self.use_cd = use_cd
        self.init_beta = init_beta
        self.learn_beta = learn_beta
        self.beta_lr_scale = beta_lr_scale
        self.instrumented = instrumented
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
        if self.instrumented:
            self.instrument_record = InstrumentRecord()
        #
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

        init_beta_driver = N.log(N.exp(self.init_beta) - 1.) / self.beta_lr_scale
        self.beta_driver = shared(N.cast[floatX] (init_beta_driver))

        self.params = [ self.W, self.c, self.vis_mean ]

        if self.learn_beta:
            self.params.append(self.beta_driver)
        #

        self.redo_theano()
    #


    def expected_energy(self, V, Q):

        """
        def f(v,q,w,beta):
            return beta * (
					#0.5 * T.dot(v,v)
					#- T.dot(self.vis_mean,v)
					#-T.dot(v- (1.-self.fold_biases)*self.vis_mean,T.dot(self.W,q))
					#+
                    #0.5*T.dot(T.dot(self.W,q).T,T.dot(self.W,q))
					#+0.5 * T.dot(w,(1.-self.fold_biases)*q-T.sqr(q))
				    #) - T.dot(self.c,q)

        rval, updates = scan( f, sequences = [V,Q], non_sequences = [self.W_norms, self.beta] )

        assert len(updates.keys()) == 0

        #return T.mean(rval)"""  #scan is really slow

        return  0.5 * self.beta * self.nvis * T.mean(T.sqr(V)) - \
                            self.beta * T.mean(T.dot(V,self.vis_mean)) - \
                            self.beta * self.nvis * T.mean( ( V - (1.-self.fold_biases)*self.vis_mean ) * T.dot(Q,self.W.T) ) + \
                            0.5 * self.beta * self.nvis * T.mean(T.sqr(T.dot(Q,self.W.T))) + \
                            0.5 * self.beta * T.mean( \
                                                     T.dot(\
                                                           (1.-self.fold_biases)*Q-T.sqr(Q),self.W_norms\
                                                          )\
                                                    ) \
                            - T.mean(T.dot(Q,self.c))

    def redo_theano(self):

        init_names = dir(self)

        if 'theano_rng' not in dir(self):
            assert self.initialized
            print "WARNING: pickle did not contain theano_rng, starting from default one"
            self.reset_rng()
            return

        self.beta = T.nnet.softplus(self.beta_driver * self.beta_lr_scale)

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

        self.hid_exp_func = function([X],pos_Q)

        if self.use_cd:
            samples = [ X ]
        else:
            samples = [ self.chains ]

        outside_pos_Q = shared(N.cast[floatX](N.zeros((1,1))))
        outside_neg_Q = shared(N.cast[floatX](N.zeros((1,1))))

        for i in xrange(self.gibbs_iters):
            if i == 0 and not self.use_cd:
                #if using SML, the first Q of gibbs sampling was already computed during the
                #previous call to learn_mini_batch
                samples.append(self.gibbs_step( Q = outside_neg_Q) )
            else:
                samples.append(self.gibbs_step( V = samples[-1]))
            #
        #

        #if using SML, this needs to be called on the first mini batch to make sure outside_neg_Q is initialized
        self.set_up_sampler = function([],updates=[(outside_neg_Q, self.infer_Q(self.chains))])
        self.first_mini_batch = True

        final_sample = samples[-1]
        final_sample.name = 'final_sample'

        neg_Q = self.infer_Q(final_sample)
        neg_Q.name = 'neg_Q'



        sampling_updates = [ (outside_pos_Q, pos_Q), (outside_neg_Q, neg_Q) ]

        if not self.use_cd:
            sampling_updates.append((self.chains,final_sample))

        self.run_sampling = function([X], updates = sampling_updates, name = 'run_sampling')
        #self.run_sampling = function([X],[pos_Q, neg_Q], updates = sampling_updates , name='run_sampling')

        #outside_pos_Q = T.matrix()
        #outside_pos_Q.name = 'outside_pos_Q'
        #outside_neg_Q = T.matrix()
        #outside_neg_Q.name = 'outside_neg_Q'

        obj = self.expected_energy(X,outside_pos_Q) \
            - self.expected_energy(self.chains,outside_neg_Q) \
            + self.enc_weight_decay * T.mean(T.sqr(self.W))


        grads = [ T.grad(obj,param) for param in self.params ]

        #self.learn_from_samples = function([X, outside_pos_Q, outside_neg_Q, alpha], updates =
        self.learn_from_samples = function([X, alpha], updates =
        [ (param, param - alpha * grad) for (param,grad)
                    in zip(self.params, grads) ] , name='learn_from_samples')

        self.recons_func = function([X], self.gibbs_step_exp(X) , name = 'recons_func')

        self.sample = function([X], self.gibbs_step(X), name = 'sample_func')

        if self.instrumented:
            self.make_instruments()
        #

        final_names = dir(self)

        self.names_to_del = [ name for name in final_names if name not in init_names ]

    def learn(self, dataset, batch_size):
        self.learn_mini_batch(dataset.get_batch_design(batch_size))


    def error_func(self, x):
        return N.square( x - self.recons_func(x)).mean()

    def record_monitoring_error(self, dataset, batch_size, batches):
        print 'running on monitoring set'
        assert self.error_record_mode == self.ERROR_RECORD_MODE_MONITORING

        w = self.W.get_value(borrow=True)
        print 'weights summary: '+str( (w.min(),w.mean(),w.max()))

        errors = []

        if self.instrumented:
            self.clear_instruments()

        for i in xrange(batches):
            x = dataset.get_batch_design(batch_size)
            error = self.error_func(x)
            errors.append( error )
            if self.instrumented:
                self.update_instruments(x)
            #
        #


        self.error_record.append( (self.examples_seen, self.batches_seen, N.asarray(errors).mean() ) )


        if self.instrumented:
            self.instrument_record.begin_report(examples_seen = self.examples_seen, batches_seen = self.batches_seen)
            self.make_instrument_report()
            self.instrument_record.end_report()
            self.clear_instruments()
        #
        print 'monitoring set done'
    #


    def recons_from_Q(self,Q):
        return self.vis_mean + T.dot(Q, self.W.T)
    #

    def recons_err_from_Q(self,Q,V):
        return T.mean(T.sqr(V-self.recons_from_Q(Q)))

    def binary_entropy(self,Q):
        mod_Q = 1e-6 + (1.-2e-6)*Q

        return -(mod_Q * T.log(Q) + (1.-mod_Q)*T.log(1.-mod_Q))

    def make_instruments(self):
        assert not self.use_cd #currently just supports PCD

        recons_outputs = []
        ave_act_outputs = []
        cond_ent_outputs = []
        neg_chains_recons_outputs = []
        neg_chains_ave_act_outputs = []
        neg_chains_cond_ent_outputs = []

        self.instrument_X = T.matrix()

        for max_iters in xrange(1,self.mean_field_iters+1):
            pos_Q = self.infer_Q(self.instrument_X, max_iters = max_iters)
            neg_Q = self.infer_Q(self.chains, max_iters = max_iters)

            recons_outputs.append(self.recons_err_from_Q(pos_Q,self.instrument_X))
            neg_chains_recons_outputs.append(self.recons_err_from_Q(neg_Q,self.chains))

            ave_act_outputs.append(T.mean(pos_Q, axis=0))
            neg_chains_ave_act_outputs.append(T.mean(neg_Q, axis=0))

            cond_ent_outputs.append(T.mean(self.binary_entropy(pos_Q),axis=0))
            neg_chains_cond_ent_outputs.append(T.mean(self.binary_entropy(neg_Q),axis=0))
        #

        self.neg_chains_recons_after_mean_field   = function([],neg_chains_recons_outputs)
        self.neg_chains_ave_act_after_mean_field  = function([],neg_chains_ave_act_outputs)
        self.neg_chains_cond_ent_after_mean_field = function([],neg_chains_cond_ent_outputs)

        self.recons_after_mean_field_func = function([self.instrument_X],recons_outputs)
        self.ave_act_after_mean_field_func = function([self.instrument_X],ave_act_outputs)
        self.cond_ent_after_mean_field_func = function([self.instrument_X],cond_ent_outputs)

        neg_chain_norms = T.sqrt(T.sum(T.sqr(self.chains),axis=1))
        self.neg_chain_norms_summary = function([], [neg_chain_norms.min(),neg_chain_norms.mean(),neg_chain_norms.max()])

        weight_norms = T.sqrt(T.sum(T.sqr(self.W),axis=0))
        self.weight_norms_summary = function([], [weight_norms.min(),weight_norms.mean(),weight_norms.max()])

        self.hid_bias_summary = function([],[self.c.min(),self.c.mean(),self.c.max()])
        self.vis_bias_summary = function([],[self.vis_mean.min(),self.vis_mean.mean(),self.vis_mean.max()])

        self.beta_func = function([],self.beta)
    #

    def clear_instruments(self):

        self.cond_ent_after_mean_field = [[] for i in xrange(self.mean_field_iters)]
        self.recons_after_mean_field = [[] for i in xrange(self.mean_field_iters)]
        self.ave_act_after_mean_field = [[] for i in xrange(self.mean_field_iters)]
    #

    def update_instruments(self, X):
        ce = self.cond_ent_after_mean_field_func(X)
        re = self.recons_after_mean_field_func(X)

        aa = self.ave_act_after_mean_field_func(X)

        for fr, to in [ (ce,self.cond_ent_after_mean_field),
                        (re, self.recons_after_mean_field),
                        (aa, self.ave_act_after_mean_field) ]:
            assert len(to) == self.mean_field_iters
            assert len(fr) == self.mean_field_iters

            for fr_elem, to_elem in zip(fr,to):
                to_elem.append(fr_elem)
            #
        #
    #


    def make_instrument_report(self):
        r = self.instrument_record

        neg_chains_recons = self.neg_chains_recons_after_mean_field()
        neg_chains_ave_act = self.neg_chains_ave_act_after_mean_field()
        neg_chains_cond_ent = self.neg_chains_cond_ent_after_mean_field()

        for i in xrange(1,self.mean_field_iters+1):
            re = N.asarray(self.recons_after_mean_field[i-1]).mean()
            r.report(('recons_err_after_mean_field',i),re)
            r.report(('neg_recons_err_after_mean_field',i),neg_chains_recons[i-1])

            aa_mat = N.asarray(self.ave_act_after_mean_field[i-1])
            assert len(aa_mat.shape) == 2
            assert aa_mat.shape[1] == self.nhid

            aa_vec = aa_mat.mean(axis=0)
            aa_min = aa_vec.min()
            aa_mean = aa_vec.mean()
            aa_max = aa_vec.max()
            naa_vec = neg_chains_ave_act[i-1]
            naa_min = naa_vec.min()
            naa_mean = naa_vec.mean()
            naa_max = naa_vec.max()
            r.report(('ave_act_after_mean_field_min',i),aa_min)
            r.report(('ave_act_after_mean_field_mean',i),aa_mean)
            r.report(('ave_act_after_mean_field_max',i),aa_max)
            r.report(('neg_ave_act_after_mean_field_min',i),naa_min)
            r.report(('neg_ave_act_after_mean_field_mean',i),naa_mean)
            r.report(('neg_ave_act_after_mean_field_max',i),naa_max)

            ce_mat = N.asarray(self.cond_ent_after_mean_field[i-1])
            assert len(ce_mat.shape) == 2
            assert ce_mat.shape[1] == self.nhid
            ce_vec = ce_mat.mean(axis=0)
            ce_min, ce_mean, ce_max = ce_vec.min(), ce_vec.mean(), ce_vec.max()
            nce_vec = neg_chains_cond_ent[i-1]
            nce_min, nce_mean, nce_max = nce_vec.min(), nce_vec.mean(), nce_vec.max()
            r.report(('cond_ent_after_mean_field_min',i),ce_min)
            r.report(('cond_ent_after_mean_field_mean',i),ce_mean)
            r.report(('cond_ent_after_mean_field_max',i),ce_max)
            r.report(('neg_cond_ent_after_mean_field_min',i),nce_min)
            r.report(('neg_cond_ent_after_mean_field_mean',i),nce_mean)
            r.report(('neg_cond_ent_after_mean_field_max',i),nce_max)
        #


        neg_chain_norms_min, neg_chain_norms_mean, neg_chain_norms_max  = self.neg_chain_norms_summary()
        r.report('neg_chain_norms_min', neg_chain_norms_min)
        r.report('neg_chain_norms_mean', neg_chain_norms_mean)
        r.report('neg_chain_norms_max', neg_chain_norms_max)

        weight_norms_min, weight_norms_mean, weight_norms_max = self.weight_norms_summary()
        r.report('weight_norms_min', weight_norms_min)
        r.report('weight_norms_mean', weight_norms_mean)
        r.report('weight_norms_max', weight_norms_max)


        hid_bias_min, hid_bias_mean, hid_bias_max = self.hid_bias_summary()
        r.report('hid_bias_min', hid_bias_min)
        r.report('hid_bias_mean', hid_bias_mean)
        r.report('hid_bias_max', hid_bias_max)

        vis_bias_min, vis_bias_mean, vis_bias_max = self.vis_bias_summary()
        r.report('vis_bias_min', vis_bias_min)
        r.report('vis_bias_mean', vis_bias_mean)
        r.report('vis_bias_max', vis_bias_max)


        r.report('beta',self.beta_func())

    #


    def reconstruct(self, x, use_noise):
        assert x.shape[0] == 1

        print 'x summary: '+str((x.min(),x.mean(),x.max()))

        #this method is mostly a hack to make the formatting work the same as denoising autoencoder
        self.truth_shared = shared(x.copy())

        if use_noise:
            self.vis_shared = shared(x.copy() + 0.15 *  N.cast[floatX](self.rng.randn(*x.shape)))
        else:
            self.vis_shared = shared(x.copy())

        self.reconstruction = self.recons_func(self.vis_shared.get_value())

        print 'recons summary: '+str((self.reconstruction.min(),self.reconstruction.mean(),self.reconstruction.max()))


    def gibbs_step_exp(self, V = None, Q = None):
        if V is not None:
            assert Q is None

            base_name = V.name

            if base_name is None:
                base_name = 'anon'

            Q = self.infer_Q(V)
        else:
            assert Q is not None

            Q_name = Q.name

            if Q_name is None:
                Q_name = 'anon'

            base_name = 'from_Q_'+Q_name
        #


        H = self.sample_hid(Q)

        H.name =  base_name + '->hid_sample'


        sample =  self.vis_mean + T.dot(H,self.W_T)

        sample.name = base_name + '->sample_expectation'

        return sample


    def gibbs_step(self, V = None, Q = None):

        if V is not None:
            assert Q is None

            base_name = V.name

            if base_name is None:
                base_name = 'anon'
            #

        else:
            assert Q is not None
            Q_name = Q.name

            if Q_name is None:
                Q_name = 'anon'
            #

            base_name = 'from_Q_'+Q_name

        #

        m = self.gibbs_step_exp(V, Q)

        assert m.dtype == floatX
        sample = self.theano_rng.normal(size = m.shape, avg = m,
                                    std = T.sqrt(1./self.beta), dtype = m.dtype)

        sample.name = base_name + '->sample'

        return sample

    def sample_hid(self, Q):
        return self.theano_rng.binomial(size = Q.shape, n = 1, p = Q,
                                dtype = Q.dtype)


    def infer_Q(self, V, max_iters = 0):

        if max_iters > 0:
            iters = min(max_iters, self.mean_field_iters)
        else:
            iters = self.mean_field_iters
        #

        base_name = V.name

        if base_name is None:
            base_name = 'anon'

        Q =  [ self.init_mean_field_step(V) ]

        no_damp = 0

        for i in xrange(iters - 1):
            damp = i + 1 < self.mean_field_iters - self.no_damp_iters
            no_damp += (damp == False)
            Q.append ( self.damped_mean_field_step(V,Q[-1] , damp ) )
        #

        if max_iters == 0:
            assert no_damp == self.no_damp_iters
        else:
            assert no_damp == max(0, self.no_damp_iters - (self.mean_field_iters - max_iters))
        #

        for i in xrange(len(Q)):
            Q[i].name = base_name + '->Q ('+str(i)+')'

        return Q[-1]

    def init_mean_field_step(self, V):
        return T.nnet.sigmoid(self.c+self.beta*
                    (T.dot(V-(1.-self.fold_biases)*self.vis_mean,self.W)
                      -(1.-self.fold_biases)*0.5*self.W_norms    ))

    def damped_mean_field_step(self, V, P, damp):

        """def f(p,w):
            return - T.dot(self.W.T,T.dot(self.W,p))+w*(p-(1.-self.fold_biases)*.5)

        interaction_term, updates = \
            scan( f, sequences  = P, non_sequences= self.W_norms)


        assert len(updates.keys()) == 0"""

        interaction_term = - T.dot(T.dot(P,self.W.T),self.W) + self.W_norms * (P-(1.-self.fold_biases)*.5)



        Q = T.nnet.sigmoid(self.c+self.beta *
                             (T.dot(V-(1.-self.fold_biases)*self.vis_mean, self.W)
                               + interaction_term
                             )
                           )

        if damp:
            return self.damping_factor * P + (1.0 - self.damping_factor) * Q
        else:
            return Q
        #
    #


    def learn_mini_batch(self, x):

        #t1 = time.time()

        if self.first_mini_batch:
            self.first_mini_batch = False
            if not self.use_cd:
                self.set_up_sampler()
            #
        #

        self.run_sampling(x)
        self.learn_from_samples(x,self.learning_rate)

        #pos_Q, neg_Q = self.run_sampling(x)
        #self.learn_from_samples(x, pos_Q, neg_Q, self.learning_rate)

        #t2 = time.time()

        #print 'batch took '+str(t2-t1)+' sec'

        self.examples_seen += x.shape[0]
        self.batches_seen += 1
    #
#

