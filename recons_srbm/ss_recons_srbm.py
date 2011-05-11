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

class SS_ReconsSRBM:
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
                learning_rate,
                irange,
                init_c, mean_field_iters,
                damping_factor,
                no_damp_iters,
                persistent_chains, init_a, init_alpha, init_beta,  gibbs_iters,
                enc_weight_decay,
                use_cd, instrumented = False):
        self.initialized = False
        self.reset_rng()
        self.nhid = nhid
        self.nvis = nvis
        self.learning_rate = learning_rate
        self.ERROR_RECORD_MODE_MONITORING = 0
        self.error_record_mode = self.ERROR_RECORD_MODE_MONITORING
        self.init_weight_mag = irange
        self.force_batch_size = 0
        self.init_c = init_c
        self.init_a = init_a
        self.init_alpha = init_alpha
        self.persistent_chains = persistent_chains
        self.mean_field_iters = mean_field_iters
        self.no_damp_iters = no_damp_iters
        self.gibbs_iters = gibbs_iters
        self.damping_factor = damping_factor
        self.enc_weight_decay = N.cast[floatX](enc_weight_decay)
        self.names_to_del = []
        self.use_cd = use_cd
        self.init_beta = init_beta
        self.instrumented = instrumented
        self.redo_everything()

    def set_error_record_mode(self, mode):
        self.error_record_mode = mode

    def set_size_from_dataset(self, dataset):
        self.nvis = dataset.get_output_dim()
        self.redo_everything()
        self.b.set_value( dataset.get_marginals(), borrow=False)
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

        self.c = shared( N.cast[floatX](N.zeros(self.nhid) + self.init_c) )
        self.c.name = 'c'

        self.b = shared( N.cast[floatX](N.zeros(self.nvis)))
        self.b.name = 'b'

        self.chains = shared ( N.cast[floatX]( N.zeros((self.persistent_chains,self.nvis))) )
        self.chains.name = 'chains'

        self.a = shared(N.cast[floatX](N.zeros(self.nhid)+self.init_a))
        self.a.name = 'a'
        self.alpha = shared(N.cast[floatX] (N.zeros(self.nhid)+self.init_alpha))
        self.alpha.name = 'alpha'
        self.beta = shared(N.cast[floatX] (N.zeros(self.nvis)+self.init_beta))
        self.beta.name = 'beta'

        self.params = [ self.W, self.a, self.b, self.c, self.alpha, self.beta ]
        self.clip = [ 0, 0, 0, 0, 1, 1 ]

        self.redo_theano()
    #


    def expected_energy(self, V, Q, Mu1):

        name = V.name
        #V = Print('V.'+V.name,attrs=['min','mean','max'])(V); V.name = name
        #Q = #Print('Q.'+V.name,attrs=['min','mean','max'])(Q)
        #Mu1 = #Print('Mu1.'+V.name,attrs=['min','mean','max'])(Mu1)

        ugly = Q*(1/self.gamma+T.sqr(Mu1)) - T.sqr(Q)*T.sqr(Mu1)
        #ugly = #Print('ugly',attrs=['shape'])(ugly)
        ugly.name = 'ugly'
        term_1 = 0.5 * T.dot(self.w, T.mean(ugly,axis=0))
        term_1.name = 'term_1'
        #term_1 = #Print('term_1')(term_1)

        recons = T.dot(Q*Mu1,self.W.T)
        #recons = #Print('recons',attrs=['shape'])(recons)
        recons.name = 'recons'
        iterm = 0.5*self.nvis*T.mean(T.sqr(recons)*self.beta)
        #iterm = #Print('iterm',attrs=['shape'])(iterm)
        #iterm = #Print('iterm')(iterm)
        iterm.name = 'iterm'

        normalized_vis = self.beta * (V-self.b)
        main_term = - self.nvis * T.mean(normalized_vis*recons)
        #main_term = #Print('main_term',attrs=['shape'])(main_term)
        #main_term = #Print('main_term')(main_term)
        normalized_vis.name = 'normalized_vis'
        #normalized_vis = #Print('normalized_vis',attrs=['shape'])(normalized_vis)
        main_term.name = 'main_term'

        S = (1-Q)*(T.sqr(self.a)/T.sqr(self.alpha)+1./self.alpha) + Q*(T.sqr(Mu1)+1./self.gamma)
        #S = #Print('S',attrs=['shape'])(S)
        #S = #Print('S.'+V.name)(S)
        S.name = 'S'

        contain_s = 0.5 * T.mean(T.dot(S,self.alpha))
        #contain_s = #Print('contain_s',attrs=['shape'])(contain_s)
        #contain_s = #Print('contain_s')(contain_s)
        contain_s.name = 'contain_s'

        vis_bias = - self.nvis * T.mean(normalized_vis)
        #vis_bias = #Print('vis_bias',attrs=['shape'])(vis_bias)
        #vis_bias = #Print('vis_bias')(vis_bias)
        vis_bias.name = 'vis_bias'

        contain_v = 0.5 * T.mean(T.dot(T.sqr(V),self.beta))
        #contain_v = #Print('contain_v',attrs=['shape'])(contain_v)
        #contain_v = #Print('contain_v')(contain_v)
        contain_v.name = 'contain_v'

        hid_bias = -T.mean(T.dot(Q,self.c))
        #hid_bias = #Print('hid_bias',attrs=['shape'])(hid_bias)
        #hid_bias = #Print('his_bias')(hid_bias)
        hid_bias.name = 'hid_bias'

        s_bias = -T.mean(T.dot(Q*Mu1+(1.-Q)*(self.a/self.alpha),self.a))
        #s_bias = #Print('s_bias',attrs=['s_bias'])(s_bias)
        #s_bias = #Print('s_bias')(s_bias)
        s_bias.name = 's_boas'

        rval =   term_1 + iterm + main_term + contain_s + vis_bias \
                + contain_v + hid_bias + s_bias
        rval.name = 'rval'

        assert len(rval.type().broadcastable) == 0

        return rval

    def redo_theano(self):

        init_names = dir(self)

        if 'theano_rng' not in dir(self):
            assert self.initialized
            print "WARNING: pickle did not contain theano_rng, starting from default one"
            self.reset_rng()
            return


        self.W_T = self.W.T

        self.w = T.sum(self.beta * T.sqr(self.W).T,axis=1)
        self.w.name = 'w'

        #self.alpha = #Print('alpha',attrs=['min','mean','max'])(self.alpha)
        #self.w = #Print('w',attrs=['min','mean','max'])(self.w)

        self.gamma = self.alpha + self.w
        #self.gamma = #Print('gamma',attrs=['min','mean','max'])(self.gamma)

        lr = T.scalar()

        X = T.matrix()
        X.name = 'X'

        pos_Q, pos_Mu1 = self.infer_Q_Mu1(X)
        pos_Q.name = 'pos_Q'
        pos_Mu1.name = 'pos_Mu1'

        self.H_exp_func = function([X],pos_Q)
        self.Mu1_func = function([X],pos_Mu1)
        self.hid_exp_func = function([X],pos_Q*pos_Mu1)

        if self.use_cd:
            samples = [ X ]
        else:
            samples = [ self.chains ]

        outside_pos_Q = shared(N.cast[floatX](N.zeros((1,1))))
        outside_neg_Q = shared(N.cast[floatX](N.zeros((1,1))))
        outside_pos_Mu1 = shared(N.cast[floatX](N.zeros((1,1))))
        outside_neg_Mu1 = shared(N.cast[floatX](N.zeros((1,1))))

        for i in xrange(self.gibbs_iters):
            if i == 0 and not self.use_cd:
                #if using SML, the first Q of gibbs sampling was already computed during the
                #previous call to learn_mini_batch
                samples.append(self.gibbs_step( Q = outside_neg_Q, Mu1 = outside_neg_Mu1) )
            else:
                samples.append(self.gibbs_step( V = samples[-1]))
            #
        #

        #if using SML, this needs to be called on the first mini batch to make sure outside_neg_Q is initialized
        first_Q, first_Mu1 = self.infer_Q_Mu1(self.chains)
        self.set_up_sampler = function([],updates=[
            (outside_neg_Q, first_Q),
            (outside_neg_Mu1, first_Mu1)])
        self.first_mini_batch = True

        final_sample = samples[-1]
        final_sample.name = 'final_sample'

        neg_Q, neg_Mu1 = self.infer_Q_Mu1(final_sample)
        neg_Q.name = 'neg_Q'
        neg_Mu1.name = 'neg_Mu1'

        sampling_updates = [ (outside_pos_Q, pos_Q), (outside_neg_Q, neg_Q),
                             (outside_pos_Mu1, pos_Mu1), (outside_neg_Mu1, neg_Mu1) ]

        if not self.use_cd:
            sampling_updates.append((self.chains,final_sample))

        self.run_sampling = function([X], updates = sampling_updates, name = 'run_sampling')

        obj = self.expected_energy(X,outside_pos_Q, outside_pos_Mu1) \
            - self.expected_energy(self.chains,outside_neg_Q, outside_neg_Mu1) \
            + self.enc_weight_decay * T.mean(T.sqr(self.W))


        grads = [ T.grad(obj,param) for param in self.params ]

        learning_updates = []

        for i in xrange(len(self.params)):
            update = self.params[i] - lr * grads[i]
            #update = #Print(self.params[i].name+' preclip',attrs=['min','mean','max'])(update)
            if self.clip[i]:
                update = T.clip(update,.1,1000)
            #
            learning_updates.append((self.params[i],update))
        #

        self.learn_from_samples = function([X, lr], updates =
        learning_updates , name='learn_from_samples')

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

        #alpha = self.alpha.get_value(borrow=True)
        beta = self.beta.get_value(borrow=True)
        #print "alpha summary: "+str( (alpha.min(),alpha.mean(),alpha.max()))
        print "beta summary: "+str( (beta.min(), beta.mean(), beta.max()))

        if N.any(N.isnan(w)):
            raise Exception("Nan")
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


    def recons_from_Q_Mu1(self,Q,Mu1):
        return self.b + T.dot(Q*Mu1, self.W.T)
    #

    def recons_err_from_Q_Mu1(self,Q,Mu1,V):
        return T.mean(T.sqr(V-self.recons_from_Q_Mu1(Q,Mu1)))

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
            pos_Q, pos_Mu1 = self.infer_Q_Mu1(self.instrument_X, max_iters = max_iters)
            neg_Q, neg_Mu1 = self.infer_Q_Mu1(self.chains, max_iters = max_iters)

            recons_outputs.append(self.recons_err_from_Q_Mu1(pos_Q,pos_Mu1,self.instrument_X))
            neg_chains_recons_outputs.append(self.recons_err_from_Q_Mu1(neg_Q,neg_Mu1,self.chains))

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
        self.vis_bias_summary = function([],[self.b.min(),self.b.mean(),self.b.max()])

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


    def gibbs_step_exp(self, V = None, Q = None, Mu1 = None):
        if V is not None:
            assert Q is None
            assert Mu1 is None

            base_name = V.name

            if base_name is None:
                base_name = 'anon'

            Q, Mu1 = self.infer_Q_Mu1(V)
        else:
            assert Q is not None
            assert Mu1 is not None

            Q_name = Q.name

            if Q_name is None:
                Q_name = 'anon'

            base_name = 'from_Q_'+Q_name
        #


        H, S = self.sample_hid(Q, Mu1)

        H.name =  base_name + '->hid_sample'


        sample =  self.b + T.dot(H*S,self.W_T)

        sample.name = base_name + '->sample_expectation'

        return sample


    def gibbs_step(self, V = None, Q = None, Mu1 = None):

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

        m = self.gibbs_step_exp(V, Q, Mu1)

        assert m.dtype == floatX
        std = T.sqrt(1./self.beta)
        #std = #Print('vis_std',attrs=['min','mean','max'])(std)
        sample = self.theano_rng.normal(size = m.shape, avg = m,
                                    std = std, dtype = m.dtype)

        sample.name = base_name + '->sample'

        return sample

    def sample_hid(self, Q, Mu1):
        H =  self.theano_rng.binomial(size = Q.shape, n = 1, p = Q,
                                dtype = Q.dtype)
        std = T.sqrt(1./self.gamma)
        #std = #Print('hid_std',attrs=['min','mean','max'])(std)
        S = self.theano_rng.normal(size = Mu1.shape, avg = Mu1, std = std, dtype = Mu1.dtype)

        return H, S

    def infer_Q_Mu1(self, V, max_iters = 0):

        if max_iters > 0:
            iters = min(max_iters, self.mean_field_iters)
        else:
            iters = self.mean_field_iters
        #

        base_name = V.name

        if base_name is None:
            base_name = 'anon'

        first_Q, first_Mu1 = self.init_mean_field_step(V)
        Q =  [ first_Q ]
        Mu1 = [ first_Mu1 ]

        no_damp = 0

        for i in xrange(iters - 1):
            damp = i + 1 < self.mean_field_iters - self.no_damp_iters
            no_damp += (damp == False)
            new_Q, new_Mu1 = self.damped_mean_field_step(V,Q[-1],Mu1[-1],damp)
            Q.append ( new_Q )
            Mu1.append( new_Mu1)
        #

        if max_iters == 0:
            assert no_damp == self.no_damp_iters
        else:
            assert self.no_damp_iters is not None
            assert self.mean_field_iters is not None
            assert max_iters is not None
            assert no_damp == max(0, self.no_damp_iters - (self.mean_field_iters - max_iters))
        #

        for i in xrange(len(Q)):
            Q[i].name = base_name + '->Q ('+str(i)+')'

        assert len(Q[-1].type().broadcastable) == 2
        assert len(Mu1[-1].type().broadcastable) == 2

        return Q[-1], Mu1[-1]

    def Q_from_A(self, A):
        assert len(A.type().broadcastable) == 2
        return T.nnet.sigmoid(0.5*(T.sqr(A)/self.gamma-T.sqr(self.a)/self.alpha)+self.c-0.5*T.log(self.gamma/self.alpha))


    def mean_field_step(self, V, P, Mu):

        assert len(V.type().broadcastable) == 2

        iterm = T.dot(T.dot(P*Mu,self.W.T*self.beta),self.W)

        normalized_V = self.beta * (V-self.b)
        main_term = T.dot(normalized_V, self.W)

        A = self.w * P*Mu - iterm + main_term + self.a
        Mu1 = A / self.gamma
        Q = self.Q_from_A( A)

        assert len(Q.type().broadcastable) == 2

        return Q, Mu1
    #


    def init_mean_field_step(self, V, damp = True):
        #return self.damped_mean_field_step(V, T.nnet.sigmoid(self.c-0.5*T.log(self.gamma/self.alpha)), self.a/self.alpha, damp)
        return self.damped_mean_field_step(V, T.zeros_like(T.dot(V,self.W)), T.zeros_like(T.dot(V,self.W)), damp)

    def damped_mean_field_step(self, V, P, Mu, damp):

        Q, Mu1 = self.mean_field_step(V,P,Mu)

        if damp:
            r_Q =  self.damping_factor * P + (1.0 - self.damping_factor) * Q
            r_Mu = self.damping_factor * Mu + (1.0-self.damping_factor) * Mu1
        else:
            r_Q = Q
            r_Mu = Mu1
        #

        assert len(r_Q.type().broadcastable) == 2

        return r_Q, r_Mu
    #

    def debug_dump(self, x):

        print "making debug dump"

        print 'x: '+str((x.min(),x.mean(),x.max()))
        W = self.W.get_value()
        print 'W: '+str((W.min(),W.mean(),W.max()))
        w = function([],self.w)()
        print 'w: '+str((w.min(),w.mean(),w.max()))
        alpha = self.alpha.get_value()
        print 'alpha: '+str((alpha.min(),alpha.mean(),alpha.max()))
        beta = self.beta.get_value()
        print 'beta: '+str((beta.min(),beta.mean(),beta.max()))
        

        prior_Q = function([],T.nnet.sigmoid(self.c-0.5*T.log(self.gamma/self.alpha)))()
        print 'prior_Q: '+str((prior_Q.min(),prior_Q.mean(),prior_Q.max()))

        prior_Mu = function([],self.a/self.alpha)()
        print 'prior_Mu: '+str((prior_Mu.min(),prior_Mu.mean(),prior_Mu.max()))


        var = T.matrix()
        var.name = 'debug_x'
        for i in xrange(1,self.mean_field_iters+1):
            outputs = self.infer_Q_Mu1(var,max_iters=i)
            Q, Mu = function([var],outputs)(x)
            print 'after '+str(i)+' mean field steps:'
            print '\tQ: '+str((Q.min(),Q.mean(),Q.max()))
            print '\tMu: '+str((Mu.min(),Mu.mean(),Mu.max()))
        #

        assert False


    def learn_mini_batch(self, x):

        #t1 = time.time()

        if self.first_mini_batch:
            self.first_mini_batch = False
            if not self.use_cd:
                self.set_up_sampler()
            #
        #

        Mu1 = self.Mu1_func(x)
        if Mu1.max() > 500.:
            self.debug_dump(x)


        #print '\nrun_sampling\n'

        self.run_sampling(x)

        #print '\nlearn_from_samples\n'

        self.learn_from_samples(x,self.learning_rate)

        #pos_Q, neg_Q = self.run_sampling(x)
        #self.learn_from_samples(x, pos_Q, neg_Q, self.learning_rate)

        #t2 = time.time()

        #print 'batch took '+str(t2-t1)+' sec'

        self.examples_seen += x.shape[0]
        self.batches_seen += 1
    #
#

