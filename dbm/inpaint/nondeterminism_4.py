from galatea.dbm.inpaint.hack import OnMonitorError
import gc
from pylearn2.datasets.binarizer import Binarizer
from pylearn2.datasets.mnist import MNIST
import galatea.dbm.inpaint.super_dbm
import galatea.dbm.inpaint.inpaint_alg
import galatea.dbm.inpaint.super_inpaint
from pylearn2.train import Train
import pylearn2.costs.cost
from pylearn2.devtools.record import RecordMode
from collections import OrderedDict


def run(replay):
    raw_train = MNIST(
        which_set="train",
        shuffle=0,
        one_hot=1,
        start=0,
        stop=50000)

    train = Binarizer(
            raw = raw_train
            )

    model = galatea.dbm.inpaint.super_dbm.SuperDBM(
            batch_size = 1250,
            niter= 6, #note: since we have to backprop through the whole thing, this does
                      #increase the memory usage
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
                   galatea.dbm.inpaint.super_dbm.DenseMaxPool (
                            detector_layer_dim= 1000,
                            pool_size= 1,
                            sparse_init= 15,
                            layer_name= 'h1',
                            init_bias= 0.
                   ),
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
                            ('train', train),
                            ('valid', Binarizer(raw=MNIST(
                                    which_set= "train",
                                    shuffle= 0,
                                    one_hot= 1,
                                    start= 50000,
                                    stop= 60000)))]
                   ),
                   line_search_mode= 'exhaustive',
                   init_alpha= [0.0256, .128, .256, 1.28, 2.56],
                   reset_alpha= 0,
                   conjugate= 1,
                   reset_conjugate= 0,
                   max_iter= 5,
                   cost= pylearn2.costs.cost.SumOfCosts(
                           costs = [
                                   galatea.dbm.inpaint.super_inpaint.SuperInpaint(
                                            both_directions = 0,
                                            noise =  0,
                                            supervised =  1,
                                            l1_act_targets = [  .06, .07, 0. ],
                                            l1_act_eps =     [  .04,  .05, 0. ],
                                            l1_act_coeffs =  [ .01,  .000, 0.  ]
                                   ),
                                   galatea.dbm.inpaint.super_dbm.DBM_WeightDecay(
                                            coeffs = [ .0000005, .0000005, .0000005 ]
                                   )
                           ]
                   ),
                   mask_gen = galatea.dbm.inpaint.super_inpaint.MaskGen (
                            drop_prob= 0.1,
                            balance= 0,
                            sync_channels= 0
                   )
            )

    train = Train(dataset=train,
            model=model,
            algorithm=algorithm,
        extensions= [
                    galatea.dbm.inpaint.hack.ErrorOnMonitor(),
            ],
    )

    try:
        train.main_loop()
        assert False # Should raise OnMonitorError
    except OnMonitorError:
        pass

    del train
    gc.collect()

run(0)
run(1)
