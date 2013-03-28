# -*- coding: utf-8 -*-
# this file is used by jobman to generate jobs
import numpy
import os
from jobman import DD

home = os.getenv('HOME')

default_config = DD({
    # theano profiling, 0 not printing
    'profile': 1,
    # specify the correct data path
    # cifar10.npz
    # curves.npz
    # mnist_6k_1k_1k.npz
    'data': '/scratch/yaoli/Exp_scratch/data/mnist_6k_1k_1k.npz',
    'verbose': 3,
    'device': 'gpu',
    # batch for computing gradient
    # 50000 for mnist, 40000 for cifar, 20000 for curves
    # gbs=mbs=ebs=cbs=200 when sgd
    'gbs': 60000,
    # batch for computing the metric, 10000 for non-sgd
    'mbs': 10000,
    # batch for evaluating the model
    # and doing line search, 10000 for non-sgd
    'ebs': 10000,
    # number of samples to consider at any time
    # 250
    'cbs': 250,
    # daa, mlp
    'model': 'daa',
    #'sgd' 'krylov' 'natNCG', 'natSGD_jacobi'
    'algo': 'natSGD_jacobi',#'krylov',
    # Gf for park metric, amari otherwise
    'type':'Gf',
    # keep it under 1000, but bigger for sgd
    'loopIters': 1000,
    # 1 is catching NaN
    'gotNaN': 0,
    'seed': 312,
    # there must not be any space between numbers below, otherwise
    # jobman raise an error
    # mlp [1000,1000,1000],
    # cifar deep [2000,1000,1000],
    # to compare:
    #------------------
    #mnist(mlp): [500,500,2000]
    #mnist(ae):[1000,500,250,30]
    #cifar(mlp): 1000, 10000
    #curves(ae):[400,200,100,50,25,5]
    'hids': '[1000,500,250,30]',
    # stop LCG till this difference is reached
    'mrtol': 1e-4,
    # damping factor for the matrix, should be fixed for natNCG
    'mreg': 45,
    # damping factor for preconditioning
    'jreg': .02,
    # NCG restart
    'resetFreq': 40,
    # max iterations of LCG
    'miters': numpy.int32(20),
    # sgd:0.03, other 0.9, 1 or 2
    'lr': 1,
    # weight initialization formula .. not very useful to change it right now
    # xavier or small
    'init' : 'xavier',
    # error cost for deep autoencoder (note Dumi and I think Martens used cross entropy for MNIST)
    'daacost' : 'cross',
    'l2norm': 1e-5,
    # numbers of linear search
    'lsIters': 80,
    # checking the validation score, keep it low, unless SGD.
    'checkFreq': 5,
    # the size krylov space
    'krylovDim': 15,
    # lbfgs steps
    'lbfgsIters': 10,
    # natNCG uses 0
    'adaptivedamp': 1,
        })
