from pylearn2.config import yaml_parse

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

        return self.cost(self.dbm, X, Y, self.drop_mask, self.drop_mask_Y)

    def get_hiddens(self):
        """

        returns: [ hidden vars connected to binary cost, hidden vars connect to softmax cost]
            (should be one theano variable each)
        """

        X = self.X
        Y = self.Y

        scratch = self.cost(self.dbm, X, Y, self.drop_mask, self.drop_mask_Y, return_locals=True)

        hiddens = scratch['final_state']

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
                nvis: 784,
                bias_from_marginals: *raw_train,
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
               },
        """

        self.cost = yaml_parse.load(cost_string)

        self.X = X
        self.Y = Y

        self._on_load_batch = self.cost.get_fixed_var_descr().on_load_batch[0]

