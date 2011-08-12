#!/bin/env python
import numpy as N
import sys
from pylearn2.utils import serial
from pylearn2.gui.patch_viewer import PatchViewer
from pylearn2.config import yaml_parse
import theano.tensor as T
from theano import function

model = serial.load(sys.argv[1])
sigma = float(sys.argv[2])
model.redo_theano()

n = model.get_input_dim()
ns = int(N.sqrt(n))

rows = 10
examplesPerRow = 5


if 'dataset_desc' not in dir(model):
    dataset = yaml_parse.load(model.dataset_yaml_src)

if dataset.view_shape()[2] == 3:
    print 'n='+str(n)
    grey_dim = n/3
    print 'grey_dim = '+str(grey_dim)
    ns = int(N.sqrt(grey_dim))
    assert ns*ns == grey_dim

p = PatchViewer((rows,examplesPerRow*5),(ns,ns), is_color = dataset.view_shape()[2] == 3)

def reshape(x):
    if dataset.view_shape()[2] == 3:
        fuckyou = [  ]
        for i in xrange(3):
            #print 'x shape :'+str(x.shape)
            #print (i*grey_dim,(i+1)*grey_dim)
            channel = x[:,i*grey_dim:(i+1)*grey_dim]
            #print 'channel shape :'+str(channel.shape)
            fuckyou.append(channel.reshape(ns,ns,1))
        return N.concatenate( fuckyou    ,axis=2)
    else:
        return x.reshape((ns,ns))

F = T.matrix()
recons_func = function([F], model.energy_function.reconstruct(F))

def reconstruct(X, use_noise):

    corrupt_X = X.copy()
    if use_noise:
        noise = N.random.randn(*corrupt_X.shape)
        scaled_noise = noise * sigma
        corrupt_X += scaled_noise
    R = recons_func(corrupt_X)

    return X, corrupt_X, R

for i in range(0,rows):
    for j in range(0, examplesPerRow):
        x = dataset.get_batch_design(1)

        p.add_patch( reshape(x),rescale=True)
        truth, noise, reconstruction = reconstruct(x, use_noise = True)

        p.add_patch(reshape(truth ),rescale=True)
        p.add_patch(reshape(noise ), rescale=True)
        p.add_patch(reshape(reconstruction) , rescale=True)
        print ( 'mse', N.square(reconstruction-truth).mean(), \
                'mae', N.abs(reconstruction-truth).mean() )
        truth, noise, reconstruction = reconstruct(x, use_noise = False)
        p.add_patch( reshape(reconstruction) , rescale=True)
p.show()
