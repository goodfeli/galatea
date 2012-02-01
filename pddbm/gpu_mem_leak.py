#script to demonstrate that theano leaks memory on the gpu

import numpy as np

yaml_src = """!obj:pylearn2.scripts.train.Train {
    "dataset": !pkl: &src "${PYLEARN2_DATA_PATH}/stl10/stl10_32x32_whitened/unsupervised.pkl",
    "model": !obj:galatea.pddbm.pddbm.PDDBM {
        "learning_rate" : .001,
        "dbm": !obj:pylearn2.models.dbm.DBM {
                "negative_chains" : 100,
                "monitor_params" : 1,
                "rbms" : [ !pkl: "/u/goodfeli/galatea/pddbm/config/stl/full/layer_2_from_C1_A.pkl" ]
        },
        "s3c": !pkl: "/u/goodfeli/galatea/pddbm/config/stl/full/layer_1_C1.pkl",
       "inference_procedure" : !obj:galatea.pddbm.pddbm.InferenceProcedure {
                "schedule" : [ ['s',1.],   ['h',1.],   ['g',0],   ['h', 0.4], ['s',0.4],
                             ['h',0.4], ['g',0],   ['h',0.4], ['s',0.4],  ['h',0.4],
                             ['g',0],   ['h',0.4], ['s',0.4], ['h', 0.4], ['g',0],
                             ['h',0.4], ['g',0],   ['h',0.4], ['s', 0.4], ['h',0.4] ],
                "monitor_kl" : 0,
                "clip_reflections" : 1,
                "rho" : 0.5
       },
       "print_interval" :  100000,
       "sub_batch" : 1,
       #"h_penalty" : 32.,
       #"g_penalties" : [200.],
       #"g_targets" : [.1],
       #"dbm_weight_decay" : [10],
    },
    "algorithm": !obj:pylearn2.training_algorithms.default.DefaultTrainingAlgorithm {
               "batch_size" : 100,
               "batches_per_iter" : 100,
               "monitoring_batches" : 1,
               "monitoring_dataset" : !obj:pylearn2.datasets.dense_design_matrix.from_dataset {
                        "dataset" : *src,
                        "num_examples" : 100
                }
        },
    "save_path": "/invalid"
}"""

from pylearn2.config import yaml_parse


model_src = """!obj:galatea.pddbm.pddbm.PDDBM {
        "learning_rate" : .001,
        "dbm": !obj:pylearn2.models.dbm.DBM {
                "negative_chains" : 100,
                "monitor_params" : 1,
                "rbms" : [ !pkl: "/u/goodfeli/galatea/pddbm/config/stl/full/layer_2_from_C1_A.pkl" ]
        },
        "s3c": !pkl: "/u/goodfeli/galatea/pddbm/config/stl/full/layer_1_C1.pkl",
       "inference_procedure" : !obj:galatea.pddbm.pddbm.InferenceProcedure {
                "schedule" : [ ['s',1.],   ['h',1.],   ['g',0],   ['h', 0.4], ['s',0.4],
                             ['h',0.4], ['g',0],   ['h',0.4], ['s',0.4],  ['h',0.4],
                             ['g',0],   ['h',0.4], ['s',0.4], ['h', 0.4], ['g',0],
                             ['h',0.4], ['g',0],   ['h',0.4], ['s', 0.4], ['h',0.4] ],
                "monitor_kl" : 0,
                "clip_reflections" : 1,
                "rho" : 0.5
       },
       "print_interval" :  100000,
       "sub_batch" : 1,
       #"h_penalty" : 32.,
       #"g_penalties" : [200.],
       #"g_targets" : [.1],
       #"dbm_weight_decay" : [10],
    }"""

model = yaml_parse.load(model_src)

X = np.random.RandomState([1,2,3]).randn(50,model.s3c.nvis)

model.learn_mini_batch(X)

#train = yaml_parse.load(yaml_src)
#train.main_loop()
