from pylearn2.config import yaml_parse
from pylearn2.devtools import disturb_mem
import sys

disturb_mem.disturb_mem()

_, replay = sys.argv
replay = int(replay)

disturb_mem.disturb_mem()
disturb_mem.disturb_mem()
disturb_mem.disturb_mem()
disturb_mem.disturb_mem()
disturb_mem.disturb_mem()
disturb_mem.disturb_mem()

yaml_src = """
!obj:pylearn2.train.Train {
    dataset:  &train !obj:pylearn2.datasets.mnist.MNIST {
        which_set: "train",
        #binarize: 1,
        one_hot: 1,
        start: 0,
        stop: 100
    },
        model: !obj:galatea.dbm.inpaint.super_dbm.MLP_Wrapper {
                        decapitate: 0,
                        super_dbm: !obj:galatea.dbm.inpaint.super_dbm.SuperDBM {
                              batch_size : 100,
                              niter: 6, #note: since we have to backprop through the whole thing, this does
                                         #increase the memory usage
                              visible_layer: !obj:galatea.dbm.inpaint.super_dbm.BinaryVisLayer {
                                nvis: 784,
                                bias_from_marginals: *train,
                              },
              hidden_layers: [
                !obj:galatea.dbm.inpaint.super_dbm.DenseMaxPool {
                        detector_layer_dim: 100,
                        pool_size: 1,
                        sparse_init: 15,
                        layer_name: 'h0',
                        init_bias: 0.
               },
                !obj:galatea.dbm.inpaint.super_dbm.DenseMaxPool {
                        detector_layer_dim: 100,
                        pool_size: 1,
                        sparse_init: 15,
                        layer_name: 'h1',
                        init_bias: 0.
               },
               !obj:galatea.dbm.inpaint.super_dbm.Softmax {
                        sparse_init: 15,
                        layer_name: 'c',
                        n_classes: 10
               }
              ]
    },
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
               termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter { max_epochs: 10 },
               theano_function_mode: !obj:pylearn2.devtools.record.RecordMode {
                        path: 'nondeterminism_3_record.txt',
                        replay: %(replay)d
               },
               line_search_mode: 'exhaustive',
               verbose_optimization: 3,
               batch_size: 100,
               set_batch_size: 1,
               updates_per_batch: 1,
               reset_alpha: 0,
               conjugate: 1,
               reset_conjugate: 0,
               monitoring_dataset: {
                                'train' : *train,
               },
               cost : !obj:galatea.dbm.inpaint.super_dbm.SuperDBM_ConditionalNLL {
               },
        },
}
""" % locals()
disturb_mem.disturb_mem()

train = yaml_parse.load(yaml_src)
disturb_mem.disturb_mem()
train.main_loop()
disturb_mem.disturb_mem()


