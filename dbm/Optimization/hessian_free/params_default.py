# -*- coding: utf-8 -*-
from jobman import DD

default_config = DD({
    # available options: mnist, curves
    'dataset': 'mnist',
    'seed': 123,
    #-------------------------------------
    # layerwise pretraining
    # number of pretraining epochs, layerwise
    'pretraining_epochs': 1,
    'pretrain_lr': 0.1,
    'top_layer_pretrain_lr': 0.001,
    # CD-k
    'k': 1,
    # not used
    'weight_decay': 0.00002,
    #--------------------------------------
    # global pretraining
    # number of global pretraining epochs
    # this only makes sense when sgd is used
    # originally 5000
    'global_pretraining_epochs': 1,
    'global_pretrain_lr': 0.02,
    'global_pretraining_batch_size': 3000,
    # or mse
    'reconstruction_cost_type': 'cross_entropy',
    # preconditioner for lcg. jacobi
    'preconditioner': 'martens',
    # hf or sgd
    'global_pretraining_optimization': 'hf',
    #---------------------------------------
    # fine tuning
    # originally 1000
    'training_epochs': 1,
    # standard or russ
    'supervised_training_type': 'russ',
    'finetune_lr': 0.1,
    #-----------------------------------------
    # minibatch size for both layerwise pretraining and finetuning
    # note that if global pretraining is sgd, then this batch_size
    # is used as well.
    'batch_size': 20,
    })
    
