#stuff pulled out of s3c because it's deprecated, but might be worth re-using someday



#deprecated methods of S3C
    @classmethod
    def solve_vhsu_needed_stats(cls):
        return set(['mean_hsv',
                    'mean_sq_s',
                    'u_stat_1',
                    'mean_sq_hs',
                    'mean_hs',
                    'mean_h',
                    'u_stat_2',
                    'mean_sq_v'])

    def solve_vhsu_from_stats(self, stats):
         #TODO: write unit test verifying that this results in zero gradient

        #Solve for W
        mean_hsv = stats.d['mean_hsv']
        half = np.cast[config.floatX](0.5)
        u_stat_1 = stats.d['u_stat_1']
        mean_sq_hs = stats.d['mean_sq_hs']
        N = np.cast[config.floatX](self.nhid)

        numer1 = mean_hsv.T
        numer2 = half * u_stat_1.T

        numer = numer1 + numer2

        #mean_sq_hs = Print('mean_sq_hs',attrs=['mean'])(mean_sq_hs)

        denom = N * mean_sq_hs

        new_W = numer / denom
        new_W.name = 'new_W'


        #Solve for mu
        mean_hs = stats.d['mean_hs']
        mean_h =  stats.d['mean_h']
        mean_h = Print('mean_h',attrs=['min','mean','max'])(mean_h)
        new_mu = mean_hs / (mean_h + self.W_eps)
        new_mu.name = 'new_mu'


        #Solve for bias_hid
        denom = T.clip(mean_h - 1., -1., -1e-10)


        new_bias_hid = T.log( - mean_h / denom )
        new_bias_hid.name = 'new_bias_hid'


        #Solve for alpha
        mean_sq_s = stats.d['mean_sq_s']
        one = np.cast[config.floatX](1.)
        two = np.cast[config.floatX](2.)
        denom = mean_sq_s + mean_h * T.sqr(new_mu) - two * new_mu * mean_hs
        new_alpha =  one / denom
        new_alpha.name = 'new_alpha'


        #Solve for B
        #new_W = Print('new_W',attrs=['mean'])(new_W)

        numer = T.sqr(N)+one
        numer = Print('numer')(numer)
        assert numer.dtype == config.floatX
        u_stat_2 = stats.d['u_stat_2']
        #u_stat_2 = Print('u_stat_2',attrs=['mean'])(u_stat_2)

        mean_sq_v = stats.d['mean_sq_v']
        #mean_sq_v = Print('mean_sq_v',attrs=['mean'])(mean_sq_v)

        mean_sq_hs = Print('mean_sq_hs',attrs=['mean'])(mean_sq_hs)
        #mean_hsv = Print('mean_hsv',attrs=['mean'])(mean_hsv)

        dotC =  T.dot(T.sqr(new_W), mean_sq_hs)
        dotC.name = 'dotC'
        denom1 = N * dotC
        denom2 = half * u_stat_2
        denom3 = - (new_W.T *  u_stat_1).sum(axis=0)
        denom4 = - two * (new_W.T * mean_hsv).sum(axis=0)
        denom5 = mean_sq_v

        denom = T.clip(denom1 + denom2 + denom3 + denom4 + denom5, 1e-8, 1e12)
        #denom = Print('denom', attrs=['min','max'])(denom)
        assert denom.dtype == config.floatX

        new_B = numer / denom
        new_B.name = 'new_B'
        assert new_B.dtype == config.floatX


        return new_W, new_bias_hid, new_alpha, new_mu, new_B

    def log_likelihood_vhsu(self, stats):

        Z_b_term = - T.nnet.softplus(self.bias_hid).sum()
        Z_alpha_term = 0.5 * T.log(self.alpha).sum()

        N = np.cast[config.floatX]( self.nhid )
        D = np.cast[config.floatX]( self.nvis )
        half = np.cast[config.floatX]( 0.5)
        one = np.cast[config.floatX](1.)
        two = np.cast[config.floatX](2.)
        four = np.cast[config.floatX](4.)
        pi = np.cast[config.floatX](np.pi)

        Z_B_term = half * (np.square(N) + one) * T.log(self.B).sum()

        Z_constant_term = - half * (N+D)*np.log(two*pi) - half * np.square(N)*D*np.log(four*pi)


        negative_log_Z = Z_b_term + Z_alpha_term + Z_B_term + Z_constant_term
        negative_log_Z.name = 'negative_log_Z'
        assert len(negative_log_Z.type.broadcastable) == 0

        u_stat_1 = stats.d['u_stat_1']

        first_term = half * T.dot(self.B, (self.W.T * u_stat_1).sum(axis=0) )

        assert len(first_term.type.broadcastable) == 0

        mean_hsv = stats.d['mean_hsv']

        second_term = T.sum(self.B *  T.sum(self.W.T * mean_hsv,axis=0), axis=0)

        assert len(second_term.type.broadcastable) == 0


        mean_sq_hs = stats.d['mean_sq_hs']
        third_term = - half * N *  T.dot(self.B, T.dot(T.sqr(self.W),mean_sq_hs))

        mean_hs = stats.d['mean_hs']

        fourth_term = T.dot(self.mu, self.alpha * mean_hs)

        mean_sq_v = stats.d['mean_sq_v']

        fifth_term = - half * T.dot(self.B, mean_sq_v)

        mean_sq_s = stats.d['mean_sq_s']

        sixth_term = - half * T.dot(self.alpha, mean_sq_s)

        mean_h = stats.d['mean_h']

        seventh_term = T.dot(self.bias_hid, mean_h)

        eighth_term = - half * T.dot(mean_h, self.alpha * T.sqr(self.mu))

        u_stat_2 = stats.d['u_stat_2']

        ninth_term = - (one / four ) * T.dot( self.B, u_stat_2)

        ne_first_quarter = first_term + second_term
        assert len(ne_first_quarter.type.broadcastable) == 0

        ne_second_quarter = third_term + fourth_term
        assert len(ne_second_quarter.type.broadcastable) ==0


        ne_first_half = ne_first_quarter + ne_second_quarter
        assert len(ne_first_half.type.broadcastable) == 0

        ne_second_half = fifth_term + sixth_term + seventh_term + eighth_term + ninth_term
        assert len(ne_second_half.type.broadcastable) == 0

        negative_energy = ne_first_half + ne_second_half
        negative_energy.name = 'negative_energy'
        assert len(negative_energy.type.broadcastable) ==0

        rval = negative_energy + negative_log_Z
        assert len(rval.type.broadcastable) == 0
        rval.name = 'log_likelihood_vhsu'

        return rval


    def log_likelihood_u_given_hs(self, stats):
        """Note: drops some constant terms """

        NH = np.cast[config.floatX](self.nhid)

        mean_sq_hs = stats.d['mean_sq_hs']
        second_hs = stats.d['second_hs']
        mean_D_sq_mean_Q_hs = stats.d['mean_D_sq_mean_Q_hs']

        term1 = 0.5 * T.sqr(NH) * T.sum(T.log(self.B))
        #term1 = Print('term1')(term1)
        term2 = 0.5 * (NH + 1) * T.dot(self.B,T.dot(self.W,mean_sq_hs))
        #term2 = Print('term2')(term2)
        term3 = - (self.B *  ( second_hs.dimshuffle('x',0,1) * self.W.dimshuffle(0,1,'x') *
                        self.W.dimshuffle(0,'x',1)).sum(axis=(1,2))).sum()
        #term3 = Print('term3')(term3)
        a = T.dot(T.sqr(self.W), mean_D_sq_mean_Q_hs)
        term4 = -0.5 * T.dot(self.B, a)
        #term4 = Print('term4')(term4)

        rval = term1 + term2 + term3 + term4

        return rval





#deprecated E steps

class VHSU_E_Step(E_step):
    """ A variational E-step that works by running mean field on
        the auxiliary variable model """

    def __init__(self, N_schedule):
        """
        parameters:
            N_schedule: list of values to use for N throughout mean field updates.
                    len(N_schedule) determines # mean field steps
        """
        self.N_schedule = N_schedule

        super(VHSU_E_Step, self).__init__()


    def init_mf_Mu1(self, V):
        #Mu1 = (self.alpha*self.mu + T.dot(V*self.B,self.W))/(self.alpha+self.w)
        #Just use the prior
        Mu1 = self.model.mu.dimshuffle('x',0)
        assert Mu1.tag.test_value != None

        Mu1.name = "init_mf_Mu1"

        return Mu1
    #


    def mean_field_H(self, U, V, NH):

        BW = self.model.W * (self.model.B.dimshuffle(0,'x'))

        filt = T.dot(V,BW)

        u_contrib = (U * BW.dimshuffle('x',1,0)).sum(axis=2)

        pre_sq = filt - u_contrib + self.model.alpha * self.model.mu

        sq_term = T.sqr(pre_sq)

        beta = self.model.alpha + NH * self.model.w

        log_term = T.log(1.0 + NH * self.model.w / self.model.alpha )

        H = T.nnet.sigmoid(-self.h_coeff() + 0.5 * sq_term / beta  - 0.5 * log_term )

        H.name = "mean_field_H"

        return H
    #

    def mean_field_Mu1(self, U, V, NH):

        beta = self.model.alpha + NH * self.model.w

        BW = self.model.W * self.model.B.dimshuffle(0,'x')

        filt = T.dot(V,BW)

        u_mod = - (U * BW.dimshuffle('x',1,0)).sum(axis=2)

        Mu1 = (filt + u_mod + self.model.alpha * self.model.mu) / beta

        Mu1.name = "mean_field_Mu1"

        return Mu1
    #


    def mean_field_Sigma1(self, NH):
        Sigma1 = 1./(self.model.alpha + NH * self.model.w)

        Sigma1.name = "mean_field_Sigma1"

        return Sigma1
    #


    def mean_field(self, V):
        alpha = self.model.alpha

        sigma0 = 1. / alpha
        mu0 = T.zeros_like(sigma0)

        H   =    self.init_mf_H(V)
        Mu1 =    self.init_mf_Mu1(V)


        for NH in self.N_schedule:
            U   = self.mean_field_U  (H = H, Mu1 = Mu1, NH = NH)
            H   = self.mean_field_H  (U = U, V = V,     NH = NH)
            Mu1 = self.mean_field_Mu1(U = U, V = V,     NH = NH)


        Sigma1 = self.mean_field_Sigma1(NH = np.cast[config.floatX](self.model.nhid))

        return {
                'H' : H,
                'mu0' : mu0,
                'Mu1' : Mu1,
                'sigma0' : sigma0,
                'Sigma1': Sigma1,
                'U' : U
                }
    #

    def mean_field_U(self, H, Mu1, NH):

        W = self.model.W

        prod = Mu1 * H

        first_term = T.dot(prod, W.T)
        first_term_broadcast = first_term.dimshuffle(0,'x',1)

        W_broadcast = W.dimshuffle('x',1,0)
        prod_broadcast = prod.dimshuffle(0,1,'x')

        second_term = NH * W_broadcast * prod_broadcast

        U = first_term_broadcast - second_term

        U.name = "mean_field_U"

        return U
    #

    def h_coeff(self):
        """ Returns the coefficient on h in the energy function """
        return - self.model.bias_hid  + 0.5 * T.sqr(self.model.mu) * self.model.alpha

    def init_mf_H(self,V):
        nhid = self.model.nhid
        w = self.model.w
        alpha = self.model.alpha
        mu = self.model.mu
        W = self.model.W
        B = self.model.B

        NH = np.cast[config.floatX] ( nhid)
        arg_to_log = 1.+(1./alpha) * NH * w

        hid_vec = alpha * mu
        #assert (hasattr(V,'__array__') or (V.tag.test_value is not None))
        dotty_thing = T.dot(V*B, W)
        pre_sq = hid_vec + dotty_thing
        numer = T.sqr(pre_sq)
        denom = alpha + w
        frac = numer/ denom

        first_term = 0.5 *  frac

        H = T.nnet.sigmoid( first_term - self.h_coeff() - 0.5 * T.log(arg_to_log) )


        #just use the prior
        H = T.nnet.sigmoid( self.model.bias_hid )

        #H = Print('init_mf_H')(H)

        return H
    #



#deprecated M steps

class VHSU_M_Step(M_Step):
    """ An M-step based on learning using the distribution over
        V,H,S, and U-- i.e. good old-fashioned, theoretically
        justified EM

        This M step has been unit tested and seems to work correctly
        in unit tests. It has not been shown to work well in learning
        experiments. That could mean the auxiliary variables are a bad
        idea or it could mean something is wrong with the VHSU E step.
    """

    def get_monitoring_channels(self, V, model):

        hidden_obs  = model.e_step.mean_field(V)

        stats = SufficientStatistics.from_observations(needed_stats = S3C.log_likelihood_vhsu_needed_stats(), X =V, \
                                                            N = np.cast[config.floatX](model.nhid),
                                                            B = model.B,
                                                            W = model.W,
                                                            **hidden_obs)

        obj = model.log_likelihood_vhsu(stats)

        return { 'log_likelihood_vhsu' : obj }


class VHSU_Solve_M_Step(VHSU_M_Step):

    def __init__(self, new_coeff):
        self.new_coeff = np.cast[config.floatX](float(new_coeff))

    def needed_stats(self):
        return S3C.solve_vhsu_needed_stats()

    def get_updates(self, model, stats):

        W, bias_hid, alpha, mu, B = model.solve_vhsu_from_stats(stats)

        learning_updates = take_step(model, W, bias_hid, alpha, mu, B, self.new_coeff)

        return learning_updates


class VHSU_Grad_M_Step(VHSU_M_Step):

    def __init__(self, learning_rate):
        self.learning_rate = np.cast[config.floatX](float(learning_rate))

    def get_updates(self, model, stats):

        params = model.get_params()

        obj = model.log_likelihood_vhsu(stats)

        grads = T.grad(obj, params, consider_constant = stats.d.values())

        updates = {}

        for param, grad in zip(params, grads):
            #if param is model.W:
            #    grad = Print('grad_W',attrs=['min','mean','max'])(grad)

            updates[param] = param + self.learning_rate * grad

        return updates



