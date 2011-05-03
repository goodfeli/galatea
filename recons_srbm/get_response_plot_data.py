import sys
from framework.utils import serial
import numpy as N

model = serial.load(sys.argv[1])
model.redo_theano()
dataset = serial.load(sys.argv[2])
X = dataset.get_design_matrix()
output_path = sys.argv[3]

batch_size = 5
nhid = model.get_output_dim()

W = model.W.get_value()
b = model.c.get_value()

print 'making dot products'
dots = N.cast['float32'](N.dot(X,W))
print 'done'

print 'making activations'
acts = N.cast['float32'](N.zeros((X.shape[0],nhid)))

for i in xrange(0,X.shape[0],batch_size):
    if i % 1000 == 0:
        print i
    cur_batch_size = min(batch_size, X.shape[0]-batch_size+1)

    acts[i:i+cur_batch_size,:] = model.hid_exp_func(X[i:i+cur_batch_size,:])

print 'saving data'

dots_path = output_path + '_dots.npy'
acts_path = output_path + '_acts.npy'

N.save(dots_path, dots)
N.save(acts_path, acts)

serial.save(output_path+'.pkl',{ 'dots' : dots_path, 'acts' : acts_path, 'b' : b } )
