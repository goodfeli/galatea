import numpy as np

from pylearn2.utils import serial

num_jobs = 25

rng = np.random.RandomState([2013, 4, 20])

for job_id in xrange(num_jobs):
    train_file_full_stem = 'exp/'+str(job_id)+'/job'

    num_mf_iter = rng.randint(5,16)
    h0_col_norm = rng.uniform(1., 5.)
    h1_col_norm = rng.uniform(1., 5.)
    y_col_norm = rng.uniform(1., 5.)

    def random_init_string():
        irange = 10. ** rng.uniform(-2.3, -1.)
        return "irange: " + str(irange)
    h0_init = random_init_string()
    h1_init = random_init_string()
    if rng.randint(2):
        y_init = "sparse_init: 0"
    else:
        y_init = random_init_string()

    h0_init_bias = 0. # rng.uniform(-3., 0.)
    h1_init_bias = 0. # rng.uniform(-3., 0.)
    learning_rate =  10. ** rng.uniform(-2., -.5)

    if rng.randint(2):
        msat = 2
    else:
        msat = rng.randint(2, 1000)

    final_momentum = rng.uniform(.5, .9)

    lr_sat = rng.randint(200, 1000)

    decay = 10. ** rng.uniform(-3, -1)

    h0_dim = rng.randint(500, 4000)
    h1_dim = rng.randint(1000, 4000)

    yaml_str = \
    """!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.mnist.MNIST {
        which_set: "train",
        shuffle: 0,
        one_hot: 1,
        start: 0,
        stop: 50000
    },
    model: !obj:pylearn2.models.mlp.MLP {
        batch_size : 100,
        nvis: 784,
        layers: [
        !obj:pylearn2.models.mlp.RectifiedLinear {
                max_col_norm: %(h0_col_norm)f,
                dim: %(h0_dim)d,
                %(h0_init)s,
                layer_name: 'h0',
                init_bias: %(h0_init_bias)f
        },
        !obj:pylearn2.models.mlp.RectifiedLinear {
                max_col_norm: %(h1_col_norm)f,
                dim: %(h1_dim)d,
                %(h1_init)s,
                layer_name: 'h1',
                init_bias: %(h1_init_bias)f
        },
        !obj:pylearn2.models.mlp.Softmax {
                max_col_norm: %(y_col_norm)f,
                %(y_init)s,
                layer_name: 'y',
                n_classes: 10
        }
        ]
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        monitoring_dataset : {
            # 'train': *train,
            valid: !obj:pylearn2.datasets.mnist.MNIST {
                                which_set: "train",
                                shuffle: 0,
                                one_hot: 1,
                                start: 50000,
                                stop: 60000
                            },
               },
         learning_rate: %(learning_rate)f,
        init_momentum: .5,
               cost: !obj:pylearn2.costs.mlp.dropout.Dropout {
                    input_include_probs: { 'h0' : .8 },
                    input_scales: { 'h0' : 1.25 }
               },
               termination_criterion: !obj:pylearn2.termination_criteria.MonitorBased
               {
                        channel_name: "valid_y_misclass",
                        N: 100,
                        prop_decrease: 0.
               }
        },
    extensions: [
                !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
                        channel_name: "valid_y_misclass",
                        save_path: "${PYLEARN2_TRAIN_FILE_FULL_STEM}_best.pkl"
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
    save_path: "${PYLEARN2_TRAIN_FILE_FULL_STEM}.pkl",
    save_freq : 1
}
    """ % locals()

    serial.mkdir('exp/' + str(job_id))
    f = open(train_file_full_stem + '.yaml', 'w')
    f.write(yaml_str)
    f.close()
