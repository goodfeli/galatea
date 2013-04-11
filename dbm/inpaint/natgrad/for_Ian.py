from pylearn2.config import yaml_parse
import numpy
import cPickle
import theano
from minres import minresQLP, minresQLP_messages
from utils import safe_clone
import theano.tensor as TT

class ThingForIan(object):


    def thing_to_pickle(self):
        return self.dbm

    def get_params(self):
        """
        Call this to get the list of shared variables to update.
        """
        return self.dbm.get_params()


    def censor_updates(self, updates):
        """
        Call this on the updates dictionary before you apply it for your
        theano function. This is to implement constrained optimization.
        """

        self.dbm.censor_updates(updates)

    def on_load_batch(self, X, Y):
        """
        X: numpy matrix of features
        Y: numpy matrix of one-hot labels
        This should be called every time you load a new batch of data.
        """
        return self._on_load_batch(X, Y)

    def get_cost(self):
        """
        """
        X = self.X
        Y = self.Y

        return self.cost(self.dbm, X, Y, drop_mask = self.drop_mask, drop_mask_Y = self.drop_mask_Y)

    def get_hiddens(self):
        """

        returns: [ hidden vars connected to binary cost, hidden vars connect to softmax cost]
            (should be one theano variable each)
        """

        X = self.X
        Y = self.Y

        for cost in self.cost.costs:
            if 'paint' in str(type(cost)):
                inpaint_cost = cost

        scratch = inpaint_cost(self.dbm, X, Y, drop_mask = self.drop_mask,
                drop_mask_Y = self.drop_mask_Y, return_locals=True)

        hiddens = scratch['final_state']['H_hat']

        H1, H2, Y = hiddens

        H1, _ = H1
        assert H1 is _

        H2, _ = H2
        assert H2 is _

        return [H1, H2]


    def __init__(self,
                 X,
                 Y,
                 batchsize=200,
                 init_damp = 5.,
                 min_damp = .001,
                 damp_ratio = 5./4.,
                 mrtol = 1e-4,
                 miters = 100,
                 trancond = 1e-4,
                 lr = .1,
                 adapt_rho = 1):
        """
        X: theano design matrix of inputs
        Y: theano design matrix of features
        batchsize: int, describing the batch size
        init_damp: float, initial damping value
        min_damp: float, minimal damping value allowed
        damp_ratio: float, ratio used to increase damping (we decrease by
                    1./ratio)
        mrtol: float, relative tolerance error for the inversion of the metric
        miters: int, maximal number of iteration for minres
        trancond: float, (ignore) threshold for switching from MinresQLP to Minres
        lr : float/shared variable; learning rate
        adapt_rho : 0 or 1, if the damping should be heuristically adapted
        """
        self.batchsize = batchsize
        self.adapt_rho = adapt_rho
        self.damp_ratio = damp_ratio
        self.min_damp = min_damp
        # From U5I4T
        dbm_string = """
!obj:galatea.dbm.inpaint.super_dbm.SuperDBM {
              batch_size : 1250,
              niter: 5, #note: since we have to backprop through the whole thing, this does
                         #increase the memory usage
              visible_layer: !obj:galatea.dbm.inpaint.super_dbm.BinaryVisLayer {
                nvis: 784
              },
              hidden_layers: [
                !obj:galatea.dbm.inpaint.super_dbm.DenseMaxPool {
                        max_col_norm: 1.9,
                        detector_layer_dim: 500,
                        pool_size: 1,
                        sparse_init: 15,
                        layer_name: 'h0',
                        init_bias: -2.
               },
                !obj:galatea.dbm.inpaint.super_dbm.DenseMaxPool {
                        max_col_norm: 3.,
                        detector_layer_dim: 1000,
                        pool_size: 1,
                        sparse_init: 15,
                        layer_name: 'h1',
                        init_bias: -2.
               },
               !obj:galatea.dbm.inpaint.super_dbm.Softmax {
                        max_col_norm: 4.,
                        sparse_init: 0,
                        layer_name: 'c',
                        n_classes: 10
               }
              ]
    }
        """

        self.dbm = yaml_parse.load(dbm_string)

        cost_string = """

!obj:pylearn2.costs.cost.SumOfCosts {
                       costs :[
                               !obj:galatea.dbm.inpaint.super_inpaint.SuperInpaint {
                                        both_directions : 0,
                                        noise : 0,
                                        supervised: 1,
                                        l1_act_targets: [  .06, .07, 0. ],
                                        l1_act_eps:     [  .04,  .05, 0. ],
                                        l1_act_coeffs:  [ .01,  .0001, 0.  ],
                                       mask_gen : !obj:galatea.dbm.inpaint.super_inpaint.MaskGen {
                                                drop_prob: 0.5,
                                                balance: 0,
                                                sync_channels: 0
                                       },
                               },
                               #!obj:galatea.dbm.inpaint.super_dbm.DBM_WeightDecay {
                               #         coeffs: [ .0000005, .0000005, .0000005 ]
                               #}
                       ]
               }
        """

        self.cost = yaml_parse.load(cost_string)

        self.X = X
        self.Y = Y

        descr = self.cost.get_fixed_var_descr(self.dbm, X, Y)

        self._on_load_batch = descr.on_load_batch[0]

        self.drop_mask = descr.fixed_vars['drop_mask']
        self.drop_mask_Y = descr.fixed_vars['drop_mask_Y']
        self.params = self.get_params()
        self.params_shape = [x.get_value(borrow=True).shape for x in
                             self.params]

        self.damping = theano.shared(numpy.float32(init_damp))

        self.gs = [theano.shared(numpy.zeros(shp, dtype=theano.config.floatX))
                   for shp in self.params_shape]
        self.rs = [theano.shared(numpy.zeros(shp, dtype=theano.config.floatX))
                   for shp in self.params_shape]

        cost = self.get_cost()
        gs = TT.grad(cost, self.params)
        self.loc_grad_fn = theano.function([self.X, self.Y],
                                           [],
                                           updates = zip(self.gs, gs),
                                           name = 'loc_fn_grad')

        ### ### ### ### ### ### ###
        self.loc_x = theano.shared(numpy.zeros((20,784), dtype='float32'))
        self.loc_y = theano.shared(numpy.zeros((20,10), dtype='float32'))

        def compute_Gv(*args):
            (hid_sig, hid_sftmax) = self.get_hiddens()
            nw_args1 = TT.Lop(hid_sig,
                             self.params,
                             TT.Rop(hid_sig,
                                    self.params,
                                    args)/((1-hid_sig)*hid_sig*self.batchsize))
            nw_args2 = TT.Lop(hid_sftmax,
                             self.params,
                             TT.Rop(hid_sftmax,
                                    self.params,
                                    args)/(hid_sftmax*self.batchsize))
            fin_vals = [x+y for x,y in zip(nw_args1, nw_args2)]
            new_vals = safe_clone(fin_vals, [self.X, self.Y], [self.loc_x,
                                                               self.loc_y])
            return new_vals, {}


        norm_grads = TT.sqrt(sum(TT.sum(x ** 2) for x in self.gs))
        self.msgs = minresQLP_messages[1:]
        rvals = minresQLP(compute_Gv,
                          [x / norm_grads for x in self.gs],
                          self.params_shape,
                          rtol=mrtol,
                          damp=self.damping,
                          maxit=miters,
                          TranCond=trancond)
        nw_rs = [x * norm_grads for x in rvals[0]]
        flag = TT.cast(rvals[1], 'int32')
        niters = rvals[2]
        rel_residual = rvals[3]
        Anorm = rvals[4]
        Acond = rvals[5]

        norm_rs_grads = TT.sqrt(sum(TT.sum(x ** 2) for x in nw_rs))
        norm_ord0 = TT.max(abs(nw_rs[0]))
        for r in nw_rs[1:]:
            norm_ord0 = TT.maximum(norm_ord0,
                                   TT.max(abs(r)))
        updates = zip(self.rs, nw_rs)
        self.compute_natural_gradients = theano.function(
            [],
            [flag, niters, rel_residual, Anorm, Acond,
             norm_grads, norm_rs_grads, norm_ord0],
            updates=updates,
            allow_input_downcast = True,
            name='compute_riemannian_gradients',
            on_unused_input='warn')

        self.loc_old_cost = theano.function(
            [self.X, self.Y], cost, name='loc_old_cost')
        new_params = [p - lr * r for p, r in zip(self.params, self.rs)]
        new_cost = safe_clone(cost,
                              self.params,
                              new_params)
        new_err = safe_clone(cost,
                             self.params,
                             new_params)
        self.loc_new_cost = theano.function(
            [self.X, self.Y], [new_cost, new_err], name='loc_new_cost')

        updates = dict(zip(self.params, new_params))
        self.censor_updates(updates)
        self.update_params = theano.function(
            [], [], updates=updates,
            name='update_params')
        old_cost = TT.scalar('old_cost')
        new_cost = TT.scalar('new_cost')
        p_norm = TT.scalar('p_norm')
        prod = sum([TT.sum(g * r) for g, r in zip(self.gs, self.rs)])
        #pnorm = TT.sqrt(sum(TT.sum(g*g) for g in self.gs)) * \
        #        TT.sqrt(sum(TT.sum(r*r) for r in self.rs))
        dist = -lr * prod
        angle = prod / p_norm
        rho = (new_cost - old_cost) / dist
        self.compute_rho = theano.function(
            [old_cost, new_cost, p_norm], [rho, dist, angle],
            name='compute_rho')

    def step(self, X_values, Y_values):
        self.loc_x.set_value(X_values, borrow=True)
        self.loc_y.set_value(Y_values, borrow=True)
        self.loc_grad_fn(X_values, Y_values)
        rvals = self.compute_natural_gradients()

        old_cost = self.loc_old_cost(X_values, Y_values)
        new_cost, error = self.loc_new_cost(X_values, Y_values)
        rho, r_g, angle = self.compute_rho(old_cost, new_cost,
                                           rvals[5]*rvals[6])

        if self.adapt_rho == 1:
            if rho < .25:
                self.damping.set_value(numpy.float32(
                   self.damping.get_value() * self.damp_ratio))
            elif rho > .75 and self.damping.get_value() > self.min_damp:
                self.damping.set_value(numpy.float32(
                        self.damping.get_value() / self.damp_ratio))

        if new_cost >= old_cost:
                print ('Variance too large on training cost!')
                self.damping.set_value(numpy.float32(
                    self.damping.get_value() + 1.))
        else:
            self.update_params()







if __name__ == '__main__':
    # Test code
    X = TT.matrix('X')
    Y = TT.matrix('Y')
    obj = ThingForIan(X,Y)
    val_x = numpy.random.uniform(size=(50,784)).astype('float32')
    val_y = numpy.random.uniform(size=(50,10)).astype('float32')
    obj.on_load_batch(val_x, val_y)
    obj.step(val_x, val_y)
    print 'all_good'
