import numpy as np

from pylearn2.utils import serial

num_jobs = 25

rng = np.random.RandomState([2013, 4, 20])

for job_id in xrange(num_jobs):
    train_file_full_stem = 'exp/'+str(job_id)+'/sup_center'

    num_mf_iter = rng.randint(5,16)
    h0_col_norm = rng.uniform(1., 5.)
    h1_col_norm = rng.uniform(1., 5.)
    y_col_norm = rng.uniform(1., 5.)

    def random_init_string():
        if rng.randint(2):
            irange = 10. ** rng.uniform(-2.3, -1.)
            return "irange: " + str(irange)
        else:
            sparse_init = rng.randint(10, 20)
            return "sparse_init: " + str(sparse_init)
        assert False
    h0_init = random_init_string()
    h1_init = random_init_string()
    if rng.randint(2):
        y_init = "sparse_init: 0"
    else:
        y_init = random_init_string()

    h0_init_bias = rng.uniform(-3., 0.)
    h1_init_bias = rng.uniform(-3., 0.)
    learning_rate =  10. ** rng.uniform(-4., -.6)

    if rng.randint(2):
        act_reg = ""
    else:
        h0_target = rng.uniform(.03, .3)
        h1_target = rng.uniform(.03, .3)
        h0_eps = rng.uniform(0, h0_target / 2.)
        h1_eps = rng.uniform(0, h1_target / 2.)
        h0_coeff = 10. ** rng.uniform(-3, -1)
        h1_coeff = 10. ** rng.uniform(-5, -3)

        act_reg = \
        """!obj:galatea.dbm.inpaint.super_dbm.MF_L1_ActCost {
            targets: [  %(h0_target)f, %(h1_target)f, 0. ],
            eps:     [  %(h0_eps)f,  %(h1_eps)f, 0. ],
            coeffs:  [ %(h0_coeff)f, %(h1_coeff)f, 0.  ],
            supervised: 0
        },""" % locals()

    num_gibbs_steps = rng.randint(5, 16)
    toronto_neg = rng.randint(2)

    if rng.randint(2):
        msat = 2
    else:
        msat = rng.randint(2, 1000)

    final_momentum = rng.uniform(.5, .9)

    lr_sat = rng.randint(200, 1000)

    decay = 10. ** rng.uniform(-3, -1)

    yaml_str = \
    """!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.binarizer.Binarizer {
      raw: &raw_train !obj:pylearn2.datasets.mnist.MNIST {
        which_set: "train",
        shuffle: 0,
        one_hot: 1,
        start: 0,
        stop: 50000
    }},
    model: !obj:galatea.dbm.inpaint.super_dbm.SpeedMonitoringDBM {
              batch_size : 100,
              niter: %(num_mf_iter)d, #note: since we have to backprop through the whole thing, this does
                         #increase the memory usage
              visible_layer: !obj:galatea.dbm.inpaint.super_dbm.BinaryVisLayer {
                nvis: 784,
                bias_from_marginals: *raw_train,
                center: 1,
              },
              hidden_layers: [
                !obj:galatea.dbm.inpaint.super_dbm.DenseMaxPool {
                    center: 1,
                        max_col_norm: %(h0_col_norm)f,
                        detector_layer_dim: 500,
                        pool_size: 1,
                        %(h0_init)s,
                        layer_name: 'h0',
                        init_bias: %(h0_init_bias)f
               },
                !obj:galatea.dbm.inpaint.super_dbm.DenseMaxPool {
                    center: 1,
                        max_col_norm: %(h1_col_norm)f,
                        detector_layer_dim: 1000,
                        pool_size: 1,
                        %(h1_init)s,
                        layer_name: 'h1',
                        init_bias: %(h1_init_bias)f
               },
               !obj:galatea.dbm.inpaint.super_dbm.Softmax {
                    center: 1,
                        max_col_norm: %(y_col_norm)f,
                        %(y_init)s,
                        layer_name: 'c',
                        n_classes: 10
               }
              ]
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        monitoring_dataset : {
            # 'train': *train,
            raw_valid: !obj:pylearn2.datasets.mnist.MNIST {
                                which_set: "train",
                                shuffle: 0,
                                one_hot: 1,
                                start: 50000,
                                stop: 60000
                            },
               },
         learning_rate: %(learning_rate)f,
        init_momentum: .5,
               cost: !obj:pylearn2.costs.cost.SumOfCosts {
                   costs :[  %(act_reg)s
                       !obj:galatea.costs.dbm.VariationalPCD_VarianceReduction {
                           supervised: 1,
                           num_chains: 100,
                           num_gibbs_steps: %(num_gibbs_steps)d,
                       }
                       ]
               },
               termination_criterion: !obj:pylearn2.termination_criteria.MonitorBased
               {
                        channel_name: "raw_valid_misclass",
                        N: 100,
                        prop_decrease: 0.
               }
        },
    extensions: [
                !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
                        channel_name: "raw_valid_misclass",
                        save_path: "%(train_file_full_stem)s_best.pkl"
                },
                !obj:pylearn2.training_algorithms.sgd.MomentumAdjustor {
                    start: 1,
                    saturate: %(msat)d,
                    final_momentum: %(final_momentum)f
                },
                !obj:pylearn2.training_algorithms.sgd.LinearDecayOverEpoch {
                    start: 1,
                    saturate: %(lr_sat)d,
                    decay_factor: %(decay)f
                }
        ],
    save_path: "%(train_file_full_stem)s.pkl",
    save_freq : 1
}
    """ % locals()

    serial.mkdir('exp/' + str(job_id))
    f = open(train_file_full_stem + '.yaml', 'w')
    f.write(yaml_str)
    f.close()
