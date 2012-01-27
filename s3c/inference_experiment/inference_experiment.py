#!/bin/env python
#arguments
#1) Path to pkl file containing s3c model
#2) Path to the yaml file describing the new E step to put into the model
#3) Path to npy file containing data to run the experiment on
#4) Batch size: number of examples to run at the same time (make sure this is the same
#       for the heuristic method and the conjugate gradient method in each condition;
#       it should only vary across conditions as needed to avoid running out of memory,
#       not across methods)
#5) Prefix for output files. Will create
#       <prefix>_ef.npy     a vector of energy functional values
#       <prefix>_timing.npy a single scalar giving the average time per example

import sys
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import numpy as np
import theano.tensor as T
from pylearn2.models.s3c import SufficientStatistics
from pylearn2.models.s3c import S3C
from theano import function
import time

model_path, e_step_path, data_path, batch_size_str, prefix = sys.argv[1:]

print 'loading model'
model = serial.load('model_path')
model.make_pseudoparams()
assert isinstance(model, S3C)

print 'building e step'
e_step = yaml_parse.load_path(e_step_path)

print 'installing e step'
model.e_step = e_step
e_step.register_model(model)

print 'loading data'
data = np.load(data_path)
m,n = data

print 'batch_size: ',batch_size_str
batch_size = int(batch_size_str)
assert m % batch_size == 0

print 'building energy functional expression'
V = T.matrix()
obs = model.get_hidden_obs(V)

needed_stats = S3C.energy_functional_needed_stats()
stats = SufficientStatistics.from_observations(needed_stats = needed_stats, V = V, ** obs)

energy_functional = model.energy_functional( stats = stats, ** obs)
assert len(energy_functional.type.broadcastable) == 1

print 'compiling energy functional theano function'
f = function([V],energy_functional)


print 'computing energy functional values'
out = np.zeros((m,),dtype='float32')
times = []

for i in xrange(0,m,batch_size):
    print '\t',i
    t1 = time.time()
    out[i:i+batch_size] = f(data[i:i+batch_size,:])
    t2 = time.time()
    times.append( (t2-t1) / float(batch_size) )

mean_time = sum(times)/float(len(times))

print 'mean_time: ',mean_time

print 'saving results'
np.save(prefix+'_ef.npy',out)
np.save(prefix+'_timing.npy', np.zeros(tuple())+mean_time)
print 'done'
