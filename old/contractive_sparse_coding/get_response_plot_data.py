#usage: python get_response_plot_data.py <model path> <dataset path> <output path>
import sys
from framework.utils import serial
import numpy as N
from theano import function

model = serial.load(sys.argv[1])
model.redo_theano()
dataset = serial.load(sys.argv[2])
X = dataset.get_design_matrix()[1:1000]
output_path = sys.argv[3]

nhid = model.get_output_dim()

W = model.pred_W.get_value()
b = model.pred_b.get_value()
g = model.pred_g.get_value()


print 'making dot products'
dots = N.cast['float32'](N.dot(X,W))
print 'done'

print 'making activations'
acts = N.cast['float32'](N.zeros((X.shape[0],nhid)))

for i in xrange(0,X.shape[0]):
    if i % 10 == 0:
        print i

    acts[i,:] = model.infer_h(X[i,:])

print 'saving data'

dots_path = output_path + '_dots.npy'
acts_path = output_path + '_acts.npy'

N.save(dots_path, dots)
N.save(acts_path, acts)

serial.save(output_path+'.pkl',{ 'dots' : dots_path, 'acts' : acts_path, 'b' : b , 'g' : g} )
