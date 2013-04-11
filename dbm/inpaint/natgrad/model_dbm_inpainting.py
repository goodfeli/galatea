from pylearn2.config import yaml_parse
import numpy
import cPickle
import theano
import theano.tensor as TT

class ThingForRazvan(object):


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


    def __init__(self, X, Y):
        """
        X: theano design matrix of inputs
        Y: theano design matrix of features
        """

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


class DBMinpainting(object):
    def __init__(self, state):
        self.X = TT.matrix('X')
        self.Y = TT.matrix('Y')
        self.mbs = state['mbs']
        self.inputs = [self.X, self.Y]
        self.dbm_class = ThingForRazvan(self.X, self.Y)
        self.train_cost = self.dbm_class.get_cost()
        self.error = self.train_cost

        (hid_sig, hid_sftmax) = self.dbm_class.get_hiddens()
        self.hid_sig = hid_sig
        self.hid_sftmax = hid_sftmax
        self.params = self.dbm_class.get_params()
        self.params_shape = [x.get_value().shape for x in self.params]
        self.callback= self.dbm_class.on_load_batch


    def Gvs(self, *args):
        # Contribution of hid_sig
        nw_args1 = TT.Lop(self.hid_sig,
                         self.params,
                         TT.Rop(self.hid_sig,
                                self.params,
                                args)/((1-self.hid_sig)*self.hid_sig*self.mbs))
        nw_args2 = TT.Lop(self.hid_sftmax,
                         self.params,
                         TT.Rop(self.hid_sftmax,
                                self.params,
                                args)/(self.hid_sftmax*self.mbs))

        return [x+y for x,y in zip(nw_args1, nw_args2)]

    def save(self, filename):
        cPickle.dump(self.dbm_class.thing_to_pickle(), open(filename, 'wb'))

if __name__ == '__main__':
    # Test code
    X = TT.matrix()
    Y = TT.matrix()
    obj = ThingForRazvan(X,Y)
    val = obj.get_cost()
    obj.get_hiddens()
    print 'all_good'
