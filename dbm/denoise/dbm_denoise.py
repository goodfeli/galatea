from pylearn2.costs.cost import UnsupervisedCost
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T
from theano.printing import Print

class DBM_Denoise_Binary(UnsupervisedCost):
    def __init__(self,
                    drop_prob,
                    n_iter,
                    balance = False,
                    reweight = True,
                    h_target = None,
                    h_penalty = None,
                    g_target = None,
                    g_penalty = None
                    ):
        self.__dict__.update(locals())
        del self.self

    def __call__(self, model, X, return_locals = False):


        model.dbm_denoise_drop_prob = self.drop_prob
        model.dbm_denoise_n_iter = self.n_iter
        model.dbm_inpaint_balance = self.balance

        dbm = model
        theano_rng = RandomStreams(20120712)

        drop_mask = theano_rng.binomial(
                size = X.shape,
                p = self.drop_prob,
                n = 1,
                dtype = X.dtype)

        if self.balance:
            flip = theano_rng.binomial(
                    size = ( X.shape[0] ,),
                    p = 0.5,
                    n = 1,
                    dtype = X.dtype).dimshuffle(0,'x')
            drop_mask = (1-drop_mask)*flip + drop_mask * (1-flip)

        assert len(dbm.rbms) == 2

        ip = dbm.inference_procedure

        X_hat = X * (1-drop_mask) + drop_mask * T.nnet.sigmoid(dbm.bias_vis)
        H_hat = ip.infer_H_hat_one_sided(
                    other_H_hat = X_hat,
                    W = dbm.W[0] * 2.,
                    b = dbm.bias_hid[0])
        G_hat = ip.infer_H_hat_one_sided(
                    other_H_hat = H_hat,
                    W = dbm.W[1],
                    b = dbm.bias_hid[-1])

        for i in xrange(self.n_iter-1):
            H_hat = ip.infer_H_hat_two_sided(
                    H_hat_below = X_hat,
                    H_hat_above = G_hat,
                    W_below = dbm.W[0],
                    W_above = dbm.W[1],
                    b = dbm.bias_hid[0])

            X_hat = X * (1-drop_mask)+drop_mask * ip.infer_H_hat_one_sided(
                        other_H_hat = H_hat,
                        W = dbm.W[0].T,
                        b = dbm.bias_vis)

            G_hat = ip.infer_H_hat_one_sided(
                        other_H_hat = H_hat,
                        W = dbm.W[1],
                        b = dbm.bias_hid[-1])


        arg_to_sigmoid = T.dot(H_hat, dbm.W[0].T) + dbm.bias_vis
        #cross entropy is
        # - ( x log y + (1-x) log (1-y) )
        # - ( x log sigmoid( z) + (1-x) log sigmoid ( - z) )
        # x softplus( - z ) + (1-x) softplus ( z )

        unmasked_cost = X * T.nnet.softplus( - arg_to_sigmoid) + (1.-X) * T.nnet.softplus( arg_to_sigmoid)

        masked_cost = drop_mask * unmasked_cost

        if self.reweight:
            #this gives equal weight to each example
            ave_cost = masked_cost.sum() / drop_mask.sum()
        else:
            #this gives equal weight to each pixel, like in the pseudolikelihood cost
            ave_cost = masked_cost.mean()

        if self.h_target is not None:
            ave_cost = ave_cost + \
                    self.h_penalty * abs(H_hat - self.h_target).mean()

        if self.g_target is not None:
            ave_cost = ave_cost + \
                    self.g_penalty * abs(G_hat - self.g_target).mean()

        if return_locals:
            return locals()

        return ave_cost
