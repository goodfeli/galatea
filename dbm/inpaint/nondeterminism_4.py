from pylearn2.datasets.binarizer import Binarizer
from pylearn2.datasets.mnist import MNIST
import galatea.dbm.inpaint.super_dbm
import galatea.dbm.inpaint.inpaint_alg
import galatea.dbm.inpaint.super_inpaint
import pylearn2.costs.cost
from pylearn2.devtools.record import RecordMode
from collections import OrderedDict


def run(replay):
    raw_train = MNIST(
        which_set="train",
        shuffle=0,
        one_hot=1,
        start=0,
        stop=2)

    train = raw_train

    model = galatea.dbm.inpaint.super_dbm.SuperDBM(
            batch_size = 2,
            niter= 2,
            visible_layer= galatea.dbm.inpaint.super_dbm.BinaryVisLayer(
                nvis= 784,
                bias_from_marginals = raw_train,
            ),
            hidden_layers= [
                galatea.dbm.inpaint.super_dbm.DenseMaxPool(
                    detector_layer_dim= 500,
                            pool_size= 1,
                            sparse_init= 15,
                            layer_name= 'h0',
                            init_bias= 0.
                   ),
                #galatea.dbm.inpaint.super_dbm.DenseMaxPool (
                #            detector_layer_dim= 1000,
                #            pool_size= 1,
                #            sparse_init= 15,
                #            layer_name= 'h1',
                #            init_bias= 0.
                #   ),
                   galatea.dbm.inpaint.super_dbm.Softmax(
                            sparse_init= 15,
                            layer_name= 'c',
                            n_classes= 10
                   )
                  ]
        )

    algorithm = galatea.dbm.inpaint.inpaint_alg.InpaintAlgorithm(
        theano_function_mode = RecordMode(
                            file_path= "nondeterminism_4.txt",
                            replay=replay
                   ),
                   monitoring_dataset = OrderedDict([
                            ('train', train)
                            ]
                   ),
                   line_search_mode= 'exhaustive',
                   init_alpha= [0.0256, .128, .256, 1.28, 2.56],
                   reset_alpha= 0,
                   conjugate= 1,
                   reset_conjugate= 0,
                   max_iter= 5,
                   cost=\
                                   galatea.dbm.inpaint.super_inpaint.SuperInpaint(
                                            both_directions = 0,
                                            noise =  0,
                                            supervised =  1,
                                   )
                   ,
                   mask_gen = galatea.dbm.inpaint.super_inpaint.MaskGen (
                            drop_prob= 0.1,
                            balance= 0,
                            sync_channels= 0
                   )
            )

    algorithm.setup(model=model, dataset=train)
    model.monitor()

    algorithm.theano_function_mode.record.f.flush()
    algorithm.theano_function_mode.record.f.close()

run(0)
run(1)
