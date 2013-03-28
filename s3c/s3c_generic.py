
class S3C_Generic(Model, Block):
    """

    If you use S3C in published work, please cite:

        Spike-and-Slab Sparse Coding for Unsupervised Feature Discovery.
        Goodfellow, I., Courville, A., & Bengio, Y. Workshop on Challenges in
        Learning Hierarchical Models. NIPS 2011.

    """


    def __init__(self, TODO nv,nh
                        irange, init_bias_hid,
                       init_B, min_B, max_B,
                       init_alpha, min_alpha, max_alpha, init_mu,
                       m_step,
                        min_bias_hid = -1e30,
                        max_bias_hid = 1e30,
                        min_mu = -1e30,
                        max_mu = 1e30,
                       e_step = None,
                        tied_B = False,
                       monitor_stats = None,
                       monitor_params = None,
                       monitor_functional = False,
                       recycle_q = 0,
                       seed = None,
                       disable_W_update = False,
                       constrain_W_norm = False,
                       monitor_norms = False,
                       random_patches_src = None,
                       local_rf_src = None,
                       local_rf_shape = None,
                       local_rf_stride = None,
                       local_rf_draw_patches = False,
                       init_unit_W = None,
                       debug_m_step = False,
                       print_interval = 10000,
                       stop_after_hack = None):
        """"
        TODO nv: # of visible units
        TODO nh: # of hidden units
        irange: (scalar) weights are initinialized ~U( [-irange,irange] )
        init_bias_hid: initial value of hidden biases (scalar or vector)
        init_B: initial value of B (scalar or vector)
        min_B, max_B: (scalar) learning updates to B are clipped to [min_B, max_B]
        init_alpha: initial value of alpha (scalar or vector)
        min_alpha, max_alpha: (scalar) learning updates to alpha are clipped to [min_alpha, max_alpha]
        init_mu: initial value of mu (scalar or vector)
        min_mu/max_mu: clip mu updates to this range.
        e_step:      An E_Step object that determines what kind of E-step to do
                        if None, assumes that the S3C model is being driven by
                        a larger model, and does not generate theano functions
                        necessary for autonomous operation
        m_step:      An M_Step object that determines what kind of M-step to do
        tied_B:         if True, use a scalar times identity for the precision on visible units.
                        otherwise use a diagonal matrix for the precision on visible units
        constrain_W_norm: if true, norm of each column of W must be 1 at all times
        init_unit_W:      if true, each column of W is initialized to have unit norm
        monitor_stats:  a list of sufficient statistics to monitor on the monitoring dataset
        monitor_params: a list of parameters to monitor TODO: push this into Model base class
        monitor_functional: if true, monitors the EM functional on the monitoring dataset
        monitor_norms: if true, monitors the norm of W at the end of each solve step, but before
                        blending with old W by new_coeff
                        This lets us see how much distortion is introduced by norm clipping
                        Note that unless new_coeff = 1, the post-solve norm monitored by this
                        flag will not be equal to the norm of the final parameter value, even
                        if no norm clipping is activated.
        recycle_q: if nonzero, initializes the e-step with the output of the previous iteration's
                    e-step. obviously this should only be used if you are using the same data
                    in each batch. when recycle_q is nonzero, it should be set to the batch size.
        disable_W_update: if true, doesn't update W (useful for experiments where you only learn the prior)
        random_patches_src: if not None, should be a dataset
                            will set W to a batch
        local_rf_src: if not None, should be a dataset
                requires the following other params:
                    local_rf_shape: a 2 tuple
                    local_rf_stride: a 2 tuple
                    local_rf_draw_patches: if true, local receptive fields are patches from local_rf_src
                                            otherwise, they're random patches
                 will initialize the weights to have only local receptive fields. (won't make a sparse
                    matrix or anything like that)
                 incompatible with random_patches_src for now
        init_unit_W:   if True, initializes weights with unit norm
        """

        Model.__init__(self)
        Block.__init__(self)

        self.debug_m_step = debug_m_step

        self.monitoring_channel_prefix = ''

        if init_unit_W is not None and not init_unit_W:
            assert not constrain_W_norm

        self.seed = seed
        self.reset_rng()
        self.irange = irange

        TODO: store equiv of nv, nh

        if random_patches_src is not None:
            raise NotImplementedError()
        elif local_rf_src is not None:
            raise NotImplementedError()
        else:
            self.init_W = None

        self.register_names_to_del(['init_W'])

        if monitor_stats is None:
            self.monitor_stats = []
        else:
            self.monitor_stats = [ elem for elem in monitor_stats ]

        if monitor_params is None:
            self.monitor_params = []
        else:
            self.monitor_params = [ elem for elem in monitor_params ]

        self.init_unit_W = init_unit_W


        self.print_interval = print_interval

        self.constrain_W_norm = constrain_W_norm

        self.stop_after_hack = stop_after_hack
        self.monitor_norms = monitor_norms
        self.disable_W_update = disable_W_update
        self.monitor_functional = monitor_functional
        self.init_bias_hid = init_bias_hid
        self.init_alpha = float(init_alpha)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.init_B = float(init_B)
        self.min_B = float(min_B)
        self.max_B = float(max_B)
        self.m_step = m_step
        self.e_step = e_step
        if e_step is None:
            self.autonomous = False
            assert not self.m_step.autonomous
            #create a non-autonomous E step
            self.e_step = E_Step(h_new_coeff_schedule = None,
                                rho = None,
                                monitor_kl = None,
                                monitor_energy_functional = None,
                                clip_reflections = None)
            assert not self.e_step.autonomous
        else:
            self.autonomous = True
            assert e_step.autonomous
            assert self.m_step.autonomous
        self.init_mu = init_mu
        self.min_mu = np.cast[config.floatX](float(min_mu))
        self.max_mu = np.cast[config.floatX](float(max_mu))
        self.min_bias_hid = float(min_bias_hid)
        self.max_bias_hid = float(max_bias_hid)
        self.recycle_q = recycle_q
        self.tied_B = tied_B

        self.redo_everything()

    def reset_rng(self):
        if self.seed is None:
            self.rng = np.random.RandomState([1.,2.,3.])
        else:
            self.rng = np.random.RandomState(self.seed)

    def redo_everything(self):

        if self.init_W is not None:
            W = self.init_W.copy()
        else:
            TODO   W = self.rng.uniform(-self.irange, self.irange, (self.nv, self.nh))

        if self.constrain_W_norm or self.init_unit_W:
            norms = numpy_norms(W)
            W /= norms

        self.W = sharedX(W, name = 'W')
        self.bias_hid = sharedX(np.zeros(self.nh)+self.init_bias_hid, name='bias_hid')
        TODO self.alpha = sharedX(np.zeros(self.nh)+self.init_alpha, name = 'alpha')
        TODO self.mu = sharedX(np.zeros(self.nh)+self.init_mu, name='mu')
        if self.tied_B:
            self.B_driver = sharedX(0.0+self.init_B, name='B')
        else:
            TODO self.B_driver = sharedX(np.zeros(self.nv)+self.init_B, name='B')

        self.test_batch_size = 2

        if self.recycle_q:
            TODO self.prev_H = sharedX(np.zeros((self.recycle_q,self.nh)), name="prev_H")
            TODO self.prev_S = sharedX(np.zeros((self.recycle_q,self.nh)), name="prev_S")

        if self.debug_m_step:
            warnings.warn('M step debugging activated-- this is only valid for certain settings, and causes a performance slowdown.')
            self.energy_functional_diff = sharedX(0.)


        if self.monitor_norms:
            TODO self.debug_norms = sharedX(np.zeros(self.nh))

        self.redo_theano()

    @classmethod
    def energy_functional_needed_stats(cls):
        return S3C.expected_log_prob_vhs_needed_stats()

    def energy_functional(self, H_hat, S_hat, var_s0_hat, var_s1_hat, stats):
        """ Returns the energy_functional for a single batch of data
            stats is assumed to be computed from and only from
            the same data points that yielded H """

        entropy_term = self.entropy_hs(H_hat = H_hat, var_s0_hat = var_s0_hat, var_s1_hat = var_s1_hat).mean()
        likelihood_term = self.expected_log_prob_vhs(stats, H_hat = H_hat, S_hat = S_hat)

        em_functional = likelihood_term + entropy_term
        assert len(em_functional.type.broadcastable) == 0

        return em_functional

    def energy_functional_batch(self, V, H_hat, S_hat, var_s0_hat, var_s1_hat):
        """ Returns the energy_functional for a single batch of data
            stats is assumed to be computed from and only from
            the same data points that yielded H """

        entropy_term = self.entropy_hs(H_hat = H_hat, var_s0_hat = var_s0_hat, var_s1_hat = var_s1_hat)
        assert len(entropy_term.type.broadcastable) == 1
        likelihood_term = self.expected_log_prob_vhs_batch(V = V, H_hat = H_hat, S_hat = S_hat, var_s0_hat = var_s0_hat, var_s1_hat = var_s1_hat)
        assert len(likelihood_term.type.broadcastable) == 1

        em_functional = likelihood_term + entropy_term
        assert len(em_functional.type.broadcastable) == 1

        return em_functional

    def set_monitoring_channel_prefix(self, prefix):
        self.monitoring_channel_prefix = prefix

    def get_monitoring_channels(self, V):
            try:
                self.compile_mode()

                if self.m_step != None:
                    rval = self.m_step.get_monitoring_channels(V, self)
                else:
                    rval = {}

                from_e_step = self.e_step.get_monitoring_channels(V)

                rval.update(from_e_step)

                if self.debug_m_step:
                    rval['m_step_diff'] = self.em_functional_diff

                monitor_stats = len(self.monitor_stats) > 0

                if monitor_stats or self.monitor_functional:

                    obs = self.get_hidden_obs(V)

                    needed_stats = set(self.monitor_stats)

                    if self.monitor_functional:
                        needed_stats = needed_stats.union(S3C.expected_log_prob_vhs_needed_stats())

                    stats = SufficientStatistics.from_observations( needed_stats = needed_stats,
                                                                V = V, ** obs )

                    H_hat = obs['H_hat']
                    S_hat = obs['S_hat']
                    var_s0_hat = obs['var_s0_hat']
                    var_s1_hat = obs['var_s1_hat']

                    if self.monitor_functional:
                        em_functional = self.em_functional(H_hat = H_hat, S_hat = S_hat, var_s0_hat = var_s0_hat,
                                var_s1_hat = var_s1_hat, stats = stats)

                        rval['em_functional'] = em_functional

                    if monitor_stats:
                        for stat in self.monitor_stats:
                            stat_val = stats.d[stat]

                            rval[stat+'_min'] = T.min(stat_val)
                            rval[stat+'_mean'] = T.mean(stat_val)
                            rval[stat+'_max'] = T.max(stat_val)
                        #end for stat
                    #end if monitor_stats
                #end if monitor_stats or monitor_functional

                if len(self.monitor_params) > 0:
                    for param in self.monitor_params:
                        param_val = getattr(self, param)


                        rval[param+'_min'] = full_min(param_val)
                        rval[param+'_mean'] = T.mean(param_val)

                        mx = full_max(param_val)
                        assert len(mx.type.broadcastable) == 0
                        rval[param+'_max'] = mx

                        if param == 'mu':
                            abs_mu = abs(self.mu)
                            rval['mu_abs_min'] = full_min(abs_mu)
                            rval['mu_abs_mean'] = T.mean(abs_mu)
                            rval['mu_abs_max'] = full_max(abs_mu)

                        if param == 'W':
                            norms = theano_norms(self.W)
                            rval['W_norm_min'] = full_min(norms)
                            rval['W_norm_mean'] = T.mean(norms)
                            rval['W_norm_max'] = T.max(norms)

                if self.monitor_norms:
                    rval['post_solve_norms_min'] = T.min(self.debug_norms)
                    rval['post_solve_norms_max'] = T.max(self.debug_norms)
                    rval['post_solve_norms_mean'] = T.mean(self.debug_norms)

                new_rval = {}

                for key in rval:
                    new_rval[self.monitoring_channel_prefix+key] = rval[key]

                rval = new_rval

                return rval
            finally:
                self.deploy_mode()


    def __call__(self, V):
        """ this is the symbolic transformation for the Block class """
        if not hasattr(self,'w'):
            self.make_pseudoparams()
        obs = self.get_hidden_obs(V)
        return obs['H_hat']

    def compile_mode(self):
        """ If any shared variables need to have batch-size dependent sizes,
        sets them all to the sizes used for interactive debugging during graph construction """
        if self.recycle_q:
            TODO self.prev_H.set_value(
                    np.cast[self.prev_H.dtype](
                        np.zeros((self.test_batch_size, self.nh)) \
                                + 1./(1.+np.exp(-self.bias_hid.get_value()))))
            TODO self.prev_S.set_value(
                    np.cast[self.prev_S.dtype](
                        np.zeros((self.test_batch_size, self.nh)) + self.mu.get_value() ) )

    def deploy_mode(self):
        """ If any shared variables need to have batch-size dependent sizes, sets them all to their runtime sizes """
        if self.recycle_q:
            self.prev_H.set_value( np.cast[self.prev_H.dtype]( np.zeros((self.recycle_q, self.nhid)) + 1./(1.+np.exp(-self.bias_hid.get_value()))))
            self.prev_S.set_value( np.cast[self.prev_S.dtype]( np.zeros((self.recycle_q, self.nhid)) + self.mu.get_value() ) )

    def get_params(self):
        return [self.W, self.bias_hid, self.alpha, self.mu, self.B_driver ]

    def energy_vhs(self, V, H, S):
        " H MUST be binary "

        h_term = - T.dot(H, self.bias_hid)
        assert len(h_term.type.broadcastable) == 1

        s_term_1 = T.dot(T.sqr(S), self.alpha)/2.
        s_term_2 = -T.dot(S * self.mu * H , self.alpha)
        #s_term_3 = T.dot(T.sqr(self.mu * H), self.alpha)/2.
        s_term_3 = T.dot(T.sqr(self.mu) * H, self.alpha) / 2.

        s_term = s_term_1 + s_term_2 + s_term_3
        #s_term = T.dot( T.sqr( S - self.mu * H) , self.alpha) / 2.
        assert len(s_term.type.broadcastable) == 1


        recons = T.dot(H*S, self.W.T)

        v_term_1 = T.dot( T.sqr(V), self.B) / 2.
        v_term_2 = T.dot( - V * recons, self.B)
        v_term_3 = T.dot( T.sqr(recons), self.B) / 2.

        v_term = v_term_1 + v_term_2 + v_term_3

        #v_term = T.dot( T.sqr( V - recons), self. B) / 2.
        assert len(v_term.type.broadcastable) == 1

        rval = h_term + s_term + v_term
        assert len(rval.type.broadcastable) == 1

        return rval

    def expected_energy_vhs(self, V, H_hat, S_hat, var_s0_hat, var_s1_hat):
        """ This is not the same as negative expected log prob,
        which includes the constant term for the log partition function """

        var_HS = H_hat * var_s1_hat + (1.-H_hat) * var_s0_hat

        half = as_floatX(.5)

        HS = H_hat * S_hat

        sq_HS = H_hat * ( var_s1_hat + T.sqr(S_hat))

        sq_S = sq_HS + (1.-H_hat)*(var_s0_hat)

        presign = T.dot(H_hat, self.bias_hid)
        presign.name = 'presign'
        h_term = - presign
        assert len(h_term.type.broadcastable) == 1

        precoeff =  T.dot(sq_S, self.alpha)
        precoeff.name = 'precoeff'
        s_term_1 = half * precoeff
        assert len(s_term_1.type.broadcastable) == 1

        presign2 = T.dot(HS, self.alpha * self.mu)
        presign2.name = 'presign2'
        s_term_2 = - presign2
        assert len(s_term_2.type.broadcastable) == 1

        s_term_3 = half * T.dot(H_hat, T.sqr(self.mu) * self.alpha)
        assert len(s_term_3.type.broadcastable) == 1

        s_term = s_term_1 + s_term_2 + s_term_3

        v_term_1 = half * T.dot(T.sqr(V),self.B)
        assert len(v_term_1.type.broadcastable) == 1

        term6_factor1 = V * self.B
        term6_factor2 = T.dot(HS, self.W.T)
        v_term_2 = - (term6_factor1 * term6_factor2).sum(axis=1)
        assert len(v_term_2.type.broadcastable) == 1

        term7_subterm1 = T.dot(T.sqr(T.dot(HS, self.W.T)), self.B)
        assert len(term7_subterm1.type.broadcastable) == 1
        #term7_subterm2 = T.dot(var_HS, self.w)
        term7_subterm2 = - T.dot( T.dot(T.sqr(HS), T.sqr(self.W.T)), self.B)
        term7_subterm3 = T.dot( T.dot(sq_HS, T.sqr(self.W.T)), self.B )

        #v_term_3 = half * (term7_subterm1 + term7_subterm2)
        v_term_3 = half * (term7_subterm1 + term7_subterm2 + term7_subterm3)
        assert len(v_term_3.type.broadcastable) == 1

        v_term = v_term_1 + v_term_2 + v_term_3

        rval = h_term + s_term + v_term

        return rval

    def entropy_h(self, H_hat):

        return entropy_binary_vector(H_hat)

    def entropy_hs(self, H_hat, var_s0_hat, var_s1_hat):

        half = as_floatX(.5)

        one = as_floatX(1.)

        two = as_floatX(2.)

        pi = as_floatX(np.pi)

        term1_plus_term2 = self.entropy_h(H_hat)
        assert len(term1_plus_term2.type.broadcastable) == 1

        term3 = T.sum( H_hat * ( half * (T.log(var_s1_hat) +  T.log(two*pi) + one )  ) , axis= 1)
        assert len(term3.type.broadcastable) == 1

        term4 = T.dot( 1.-H_hat, half * (T.log(var_s0_hat) +  T.log(two*pi) + one ))
        assert len(term4.type.broadcastable) == 1


        for t12, t3, t4 in get_debug_values(term1_plus_term2, term3, term4):
            debug_assert(not np.any(np.isnan(t12)))
            debug_assert(not np.any(np.isnan(t3)))
            debug_assert(not np.any(np.isnan(t4)))

        rval = term1_plus_term2 + term3 + term4

        assert len(rval.type.broadcastable) == 1

        return rval

    def get_hidden_obs(self, V, return_history = False):

        return self.e_step.variational_inference(V, return_history)

    def make_learn_func(self, V):
        """
        V: a symbolic design matrix
        """

        #E step
        hidden_obs = self.get_hidden_obs(V)

        stats = SufficientStatistics.from_observations(needed_stats = self.m_step.needed_stats(),
                V = V, **hidden_obs)

        H_hat = hidden_obs['H_hat']
        S_hat = hidden_obs['S_hat']

        learning_updates = self.m_step.get_updates(self, stats, H_hat, S_hat)

        if self.recycle_q:
            learning_updates[self.prev_H] = H_hat
            learning_updates[self.prev_S] = S_hat

        self.censor_updates(learning_updates)

        if self.debug_m_step:
            em_functional_before = self.em_functional(H = hidden_obs['H'],
                                                      var_s0_hat = hidden_obs['var_s0_hat'],
                                                      var_s1_hat = hidden_obs['var_s1_hat'],
                                                      stats = stats)

            tmp_bias_hid = self.bias_hid
            tmp_mu = self.mu
            tmp_alpha = self.alpha
            tmp_W = self.W
            tmp_B_driver = self.B_driver

            self.bias_hid = learning_updates[self.bias_hid]
            self.mu = learning_updates[self.mu]
            self.alpha = learning_updates[self.alpha]
            if self.W in learning_updates:
                self.W = learning_updates[self.W]
            self.B_driver = learning_updates[self.B_driver]
            self.make_pseudoparams()

            try:
                em_functional_after  = self.em_functional(H_hat = hidden_obs['H_hat'],
                                                          var_s0_hat = hidden_obs['var_s0_hat'],
                                                          var_s1_hat = hidden_obs['var_s1_hat'],
                                                          stats = stats)
            finally:
                self.bias_hid = tmp_bias_hid
                self.mu = tmp_mu
                self.alpha = tmp_alpha
                self.W = tmp_W
                self.B_driver = tmp_B_driver
                self.make_pseudoparams()

            em_functional_diff = em_functional_after - em_functional_before

            learning_updates[self.em_functional_diff] = em_functional_diff



        print "compiling function..."
        t1 = time.time()
        rval = function([V], updates = learning_updates)
        t2 = time.time()
        print "... compilation took "+str(t2-t1)+" seconds"
        print "graph size: ",len(rval.maker.env.toposort())

        return rval

    def censor_updates(self, updates):

        assert self.bias_hid in self.censored_updates

        def should_censor(param):
            return param in updates and updates[param] not in self.censored_updates[param]

        if should_censor(self.W):
            if self.disable_W_update:
                del updates[self.W]
            elif self.constrain_W_norm:
                norms = theano_norms(updates[self.W])
                updates[self.W] /= norms.dimshuffle('x',0)

        if should_censor(self.alpha):
            updates[self.alpha] = T.clip(updates[self.alpha],self.min_alpha,self.max_alpha)

        if should_censor(self.mu):
            updates[self.mu] = T.clip(updates[self.mu],self.min_mu,self.max_mu)

        if should_censor(self.B_driver):
            updates[self.B_driver] = T.clip(updates[self.B_driver],self.min_B,self.max_B)

        if should_censor(self.bias_hid):
            updates[self.bias_hid] = T.clip(updates[self.bias_hid],self.min_bias_hid,self.max_bias_hid)

        model_params = self.get_params()
        for param in updates:
            if param in model_params:
                self.censored_updates[param] = self.censored_updates[param].union(set([updates[param]]))


    def random_design_matrix(self, batch_size, theano_rng,
                            H_sample = None):
        """
            H_sample: a matrix of values of H
                      if none is provided, samples one from the prior
                      (H_sample is used if you want to see what samples due
                        to specific hidden units look like, or when sampling
                        from a larger model that s3c is part of)
        """

        if not hasattr(self,'p'):
            self.make_pseudoparams()

        hid_shape = (batch_size, self.nhid)

        if H_sample is None:
            H_sample = theano_rng.binomial( size = hid_shape, n = 1, p = self.p)

        if hasattr(H_sample,'__array__'):
            assert len(H_sample.shape) == 2
        else:
            assert len(H_sample.type.broadcastable) == 2

        pos_s_sample = theano_rng.normal( size = hid_shape, avg = self.mu, std = T.sqrt(1./self.alpha) )

        final_hs_sample = H_sample * pos_s_sample

        assert len(final_hs_sample.type.broadcastable) == 2

        V_mean = T.dot(final_hs_sample, self.W.T)

        warnings.warn('showing conditional means (given sampled h and s) on visible units rather than true samples')
        return V_mean

        V_sample = theano_rng.normal( size = V_mean.shape, avg = V_mean, std = self.B)

        return V_sample


    @classmethod
    def expected_log_prob_vhs_needed_stats(cls):
        h = S3C.expected_log_prob_h_needed_stats()
        s = S3C.expected_log_prob_s_given_h_needed_stats()
        v = S3C.expected_log_prob_v_given_hs_needed_stats()

        union = h.union(s).union(v)

        return union


    def expected_log_prob_vhs(self, stats, H_hat, S_hat):

        expected_log_prob_v_given_hs = self.expected_log_prob_v_given_hs(stats, H_hat = H_hat, S_hat = S_hat)
        expected_log_prob_s_given_h  = self.expected_log_prob_s_given_h(stats)
        expected_log_prob_h          = self.expected_log_prob_h(stats)

        rval = expected_log_prob_v_given_hs + expected_log_prob_s_given_h + expected_log_prob_h

        assert len(rval.type.broadcastable) == 0

        return rval

    def expected_log_prob_vhs_batch(self, V, H_hat, S_hat, var_s0_hat, var_s1_hat):

        half = as_floatX(0.5)
        two = as_floatX(2.)
        pi = as_floatX(np.pi)
        N = as_floatX(self.nhid)

        #log partition function terms
        term1 = half * T.sum(T.log(self.B))
        term2 = - half * N * T.log(two * pi)
        term3 = half * T.log( self.alpha ).sum()
        term4 = - half * N * T.log(two*pi)
        term5 = - T.nnet.softplus(self.bias_hid).sum()

        negative_log_partition_function = term1 + term2 + term3 + term4 + term5
        assert len(negative_log_partition_function.type.broadcastable) == 0

        #energy term
        negative_energy = - self.expected_energy_vhs(V = V, H_hat = H_hat, S_hat = S_hat, var_s0_hat = var_s0_hat, var_s1_hat = var_s1_hat)
        assert len(negative_energy.type.broadcastable) == 1

        rval = negative_log_partition_function + negative_energy

        return rval


    def log_prob_v_given_hs(self, V, H, S):
        """
        V, H, S are SAMPLES   (i.e., H must be LITERALLY BINARY)
        Return value is a vector, of length batch size
        """

        half = as_floatX(0.5)
        two = as_floatX(2.)
        pi = as_floatX(np.pi)
        N = as_floatX(self.nhid)

        term1 = half * T.sum(T.log(self.B))
        term2 = - half * N * T.log(two * pi)

        mean_HS = H * S
        recons = T.dot(H*S, self.W.T)
        residuals = V - recons


        term3 = - half * T.dot(T.sqr(residuals), self.B)

        rval = term1 + term2 + term3

        assert len(rval.type.broadcastable) == 1

        return rval

    @classmethod
    def expected_log_prob_v_given_hs_needed_stats(cls):
        return set(['mean_sq_v','mean_hsv', 'mean_sq_hs', 'mean_sq_mean_hs'])

    def expected_log_prob_v_given_hs(self, stats, H_hat, S_hat):
        """
        Return value is a SCALAR-- expectation taken across batch index too
        """


        """
        E_v,h,s \sim Q log P( v | h, s)
        = sum_k [  E_v,h,s \sim Q log sqrt(B/2 pi) exp( - 0.5 B (v- W[v,:] (h*s) )^2)   ]
        = sum_k [ E_v,h,s \sim Q 0.5 log B_k - 0.5 log 2 pi - 0.5 B_k v_k^2 + v_k B_k W[k,:] (h*s) - 0.5 B_k sum_i sum_j W[k,i] W[k,j] h_i s_i h_j s_j ]
        = sum_k [ 0.5 log B_k - 0.5 log 2 pi - 0.5 B_k v_k^2 + v_k B_k W[k,:] (h*s) ] - 0.5  sum_k B_k sum_i,j W[k,i] W[k,j]  < h_i s_i  h_j s_j >
        = sum_k [ 0.5 log B_k - 0.5 log 2 pi - 0.5 B_k v_k^2 + v_k B_k W[k,:] (h*s) ] - (1/2T)  sum_k B_k sum_i,j W[k,i] W[k,j]  sum_t <h_it s_it  h_jt s_t>
        = sum_k [ 0.5 log B_k - 0.5 log 2 pi - 0.5 B_k v_k^2 + v_k B_k W[k,:] (h*s) ] - (1/2T)  sum_k B_k sum_t sum_i,j W[k,i] W[k,j] <h_it s_it  h_jt s_t>
        = sum_k [ 0.5 log B_k - 0.5 log 2 pi - 0.5 B_k v_k^2 + v_k B_k W[k,:] (h*s) ]
          - (1/2T)  sum_k B_k sum_t sum_i W[k,i]  sum_{j\neq i} W[k,j] <h_it s_it>  <h_jt s_t>
          - (1/2T) sum_k B_k sum_t sum_i W[k,i]^2 <h_it s_it^2>
        = sum_k [ 0.5 log B_k - 0.5 log 2 pi - 0.5 B_k v_k^2 + v_k B_k W[k,:] (h*s) ]
          - (1/2T)  sum_k B_k sum_t sum_i W[k,i] <h_it s_it> sum_j W[k,j]  <h_jt s_t>
          + (1/2T) sum_k B_k sum_t sum_i W[k,i]^2 <h_it s_it>^2
          - (1/2T) sum_k B_k sum_t sum_i W[k,i]^2 <h_it s_it^2>
        = sum_k [ 0.5 log B_k - 0.5 log 2 pi - 0.5 B_k v_k^2 + v_k B_k W[k,:] (h*s) ]
          - (1/2T)  sum_k B_k sum_t sum_i W[k,i] <h_it s_it> sum_j W[k,j]  <h_jt s_t>
          + (1/2T) sum_k B_k sum_t sum_i W[k,i]^2 (<h_it s_it>^2 - <h_it s_it^2>)
        = sum_k [ 0.5 log B_k - 0.5 log 2 pi - 0.5 B_k v_k^2 + v_k B_k W[k,:] (h*s) ]
          - (1/2T)  sum_k B_k sum_t sum_i W_ki HS_it sum_j W_kj  HS_tj
          + (1/2T) sum_k B_k sum_t sum_i sq(W)_ki ( sq(HS)-sq_HS)_it
        = sum_k [ 0.5 log B_k - 0.5 log 2 pi - 0.5 B_k v_k^2 + v_k B_k W[k,:] (h*s) ]
          - (1/2T)  sum_k B_k sum_t sum_i W_ki HS_it sum_j W_kj  HS_tj
          + (1/2T) sum_k B_k sum_t sum_i sq(W)_ki ( sq(HS)-sq_HS)_it
        = sum_k [ 0.5 log B_k - 0.5 log 2 pi - 0.5 B_k v_k^2 + v_k B_k W[k,:] (h*s) ]
          - (1/2T)  sum_k B_k sum_t sum_i W_ki HS_it sum_j W_kj  HS_tj
          + (1/2T) sum_k B_k sum_t sum_i sq(W)_ki ( sq(HS)-sq_HS)_it
        = sum_k [ 0.5 log B_k - 0.5 log 2 pi - 0.5 B_k v_k^2 + v_k B_k W[k,:] (h*s) ]
          - (1/2T)  sum_k B_k sum_t (HS_t: W_k:^T)  (HS_t:  W_k:^T)
          + (1/2) sum_k B_k  sum_i sq(W)_ki ( mean_sq_mean_hs-mean_sq_hs)_i
        = sum_k [ 0.5 log B_k - 0.5 log 2 pi - 0.5 B_k v_k^2 + v_k B_k W[k,:] (h*s) ]
          - (1/2T)  sum_t sum_k B_k  (HS_t: W_k:^T)^2
          + (1/2) sum_k B_k  sum_i sq(W)_ki ( mean_sq_mean_hs-mean_sq_hs)_i
        = sum_k [ 0.5 log B_k - 0.5 log 2 pi - 0.5 B_k v_k^2 + v_k B_k W[k,:] (h*s) ]
          - (1/2)  mean(   (HS W^T)^2 B )
          + (1/2) sum_k B_k  sum_i sq(W)_ki ( mean_sq_mean_hs-mean_sq_hs)_i
        """


        half = as_floatX(0.5)
        two = as_floatX(2.)
        pi = as_floatX(np.pi)
        N = as_floatX(self.nhid)

        mean_sq_v = stats.d['mean_sq_v']
        mean_hsv  = stats.d['mean_hsv']
        mean_sq_mean_hs = stats.d['mean_sq_mean_hs']
        mean_sq_hs = stats.d['mean_sq_hs']

        term1 = half * T.sum(T.log(self.B))
        term2 = - half * N * T.log(two * pi)
        term3 = - half * T.dot(self.B, mean_sq_v)
        term4 = T.dot(self.B , (self.W * mean_hsv.T).sum(axis=1))

        HS = H_hat * S_hat
        recons = T.dot(HS, self.W.T)
        sq_recons = T.sqr(recons)
        weighted = T.dot(sq_recons, self.B)
        assert len(weighted.type.broadcastable) == 1
        term5 = - half * T.mean( weighted)

        term6 = half * T.dot(self.B, T.dot(T.sqr(self.W), mean_sq_mean_hs - mean_sq_hs))

        rval = term1 + term2 + term3 + term4 + term5 + term6

        assert len(rval.type.broadcastable) == 0

        return rval


    @classmethod
    def expected_log_prob_s_given_h_needed_stats(cls):
        return set(['mean_h','mean_hs','mean_sq_s'])

    def expected_log_prob_s_given_h(self, stats):

        """
        E_h,s\sim Q log P(s|h)
        = E_h,s\sim Q log sqrt( alpha / 2pi) exp(- 0.5 alpha (s-mu h)^2)
        = E_h,s\sim Q log sqrt( alpha / 2pi) - 0.5 alpha (s-mu h)^2
        = E_h,s\sim Q  0.5 log alpha - 0.5 log 2 pi - 0.5 alpha s^2 + alpha s mu h + 0.5 alpha mu^2 h^2
        = E_h,s\sim Q 0.5 log alpha - 0.5 log 2 pi - 0.5 alpha s^2 + alpha mu h s + 0.5 alpha mu^2 h
        = 0.5 log alpha - 0.5 log 2 pi - 0.5 alpha mean_sq_s + alpha mu mean_hs - 0.5 alpha mu^2 mean_h
        """

        mean_h = stats.d['mean_h']
        mean_sq_s = stats.d['mean_sq_s']
        mean_hs = stats.d['mean_hs']

        half = as_floatX(0.5)
        two = as_floatX(2.)
        N = as_floatX(self.nhid)
        pi = as_floatX(np.pi)

        term1 = half * T.log( self.alpha ).sum()
        term2 = - half * N * T.log(two*pi)
        term3 = - half * T.dot( self.alpha , mean_sq_s )
        term4 = T.dot(self.mu*self.alpha,mean_hs)
        term5 = - half * T.dot(T.sqr(self.mu), self.alpha * mean_h)

        rval = term1 + term2 + term3 + term4 + term5

        assert len(rval.type.broadcastable) == 0

        return rval

    @classmethod
    def expected_log_prob_h_needed_stats(cls):
        return set(['mean_h'])

    def expected_log_prob_h(self, stats):
        """ Returns the expected log probability of the vector h
            under the model when the data is drawn according to
            stats
        """

        """
            E_h\sim Q log P(h)
            = E_h\sim Q log exp( bh) / (1+exp(b))
            = E_h\sim Q bh - softplus(b)
        """

        mean_h = stats.d['mean_h']

        term1 = T.dot(self.bias_hid, mean_h)
        term2 = - T.nnet.softplus(self.bias_hid).sum()

        rval = term1 + term2

        assert len(rval.type.broadcastable) == 0

        return rval


    def make_pseudoparams(self):
        if self.tied_B:
            #can't just use a dimshuffle; dot products involving B won't work
            #and because doing it this way makes the partition function multiply by nv automatically
            TODO self.B = self.B_driver + as_floatX(np.zeros(self.nv))
            self.B.name = 'S3C.tied_B'
        else:
            self.B = self.B_driver

        self.w = T.dot(self.B, T.sqr(self.W))
        self.w.name = 'S3C.w'

        self.p = T.nnet.sigmoid(self.bias_hid)
        self.p.name = 'S3C.p'

    def reset_censorship_cache(self):

        self.censored_updates = {}
        self.register_names_to_del(['censored_updates'])
        for param in self.get_params():
            self.censored_updates[param] = set([])

    def redo_theano(self):

        self.reset_censorship_cache()

        if not self.autonomous:
            return

        try:
            self.compile_mode()
            init_names = dir(self)

            self.make_pseudoparams()

            self.e_step.register_model(self)

            self.get_B_value = function([], self.B)

            X = T.matrix(name='V')
            #TODO X.tag.test_value = np.cast[config.floatX](self.rng.randn(self.test_batch_size,self.nv))

            self.learn_func = self.make_learn_func(X)

            final_names = dir(self)

            self.register_names_to_del([name for name in final_names if name not in init_names])
        finally:
            self.deploy_mode()

    def learn(self, dataset, batch_size):
        if self.stop_after_hack is not None:
            if self.monitor.examples_seen > self.stop_after_hack:
                print 'stopping due to too many examples seen'
                quit(-1)


        self.learn_mini_batch(dataset.get_batch_design(batch_size))
    #

    def print_status(self):
            print ""
            b = self.bias_hid.get_value(borrow=True)
            assert not np.any(np.isnan(b))
            p = 1./(1.+np.exp(-b))
            print 'p: ',(p.min(),p.mean(),p.max())
            B = self.B_driver.get_value(borrow=True)
            assert not np.any(np.isnan(B))
            print 'B: ',(B.min(),B.mean(),B.max())
            mu = self.mu.get_value(borrow=True)
            assert not np.any(np.isnan(mu))
            print 'mu: ',(mu.min(),mu.mean(),mu.max())
            alpha = self.alpha.get_value(borrow=True)
            assert not np.any(np.isnan(alpha))
            print 'alpha: ',(alpha.min(),alpha.mean(),alpha.max())
            W = self.W.get_value(borrow=True)
            assert not np.any(np.isnan(W))
            assert not np.any(np.isinf(W))
            print 'W: ',(W.min(),W.mean(),W.max())
            norms = numpy_norms(W)
            print 'W norms:',(norms.min(),norms.mean(),norms.max())

    def learn_mini_batch(self, X):

        self.learn_func(X)

        if self.monitor.examples_seen % self.print_interval == 0:
            self.print_status()

        if self.debug_m_step:
            if self.em_functional_diff.get_value() < 0.0:
                warnings.warn( "m step decreased the em functional" )
                if self.debug_m_step != 'warn':
                    quit(-1)

    #

    def get_weights_format(self):
        return ['v','h']
