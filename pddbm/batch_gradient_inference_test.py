#arg1: model to evaluate
#arg2: batch size
#arg3: # batches

import sys
import numpy as np
from pylearn2.utils import sharedX
from theano import config
from galatea.pddbm.batch_gradient_inference import BatchGradientInference

model_path = sys.argv[1]

from pylearn2.utils import serial
from pylearn2.config import yaml_parse

print 'loading model...'
model = serial.load(model_path)
print 'done'

if len(sys.argv) > 2:
    batch_size = int(sys.argv[2])
else:
    batch_size = 100

if len(sys.argv) > 3:
    num_batches = int(sys.argv[3])
else:
    num_batches = 1

print 'loading dataset...'
dataset = yaml_parse.load(model.dataset_yaml_src)
print 'done'

tester = BatchGradientInference(model)


for i in xrange(num_batches):
    X = dataset.get_batch_design(batch_size)

    result = tester.run_inference(X)

    orig_H = result['orig_H']
    orig_S = result['orig_S']
    orig_G = result['orig_G']
    H = result['H']
    S = result['S']
    G = result['G']
    orig_kl = result['orig_kl']
    kl = result['kl']

    H_dist = np.sqrt( np.square( H - orig_H ).sum() )
    S_dist = np.sqrt( np.square( S - orig_S ).sum() )
    G_dist = np.sqrt( sum( [ np.square( G_elem - G_orig_elem ).sum() for G_elem, G_orig_elem in zip(G, orig_G) ] ) )

    print 'kl improved by ',orig_kl-kl

    print 'H moved ',H_dist
    print 'S moved ',S_dist
    print 'G moved ',G_dist

    H_diffs = np.abs(H - orig_H)
    S_diffs = np.abs(S - orig_S)
    G_diffs = [ np.abs( G_elem - orig_G_elem) for G_elem, orig_G_elem in zip(G, orig_G) ]

    print 'H diffs ',(H_diffs.min(), H_diffs.mean(), H_diffs.max() )
    print 'S diffs ',(S_diffs.min(), S_diffs.mean(), S_diffs.max() )
    print 'G diffs ',( min([G_diff_elem.min() for G_diff_elem in G_diffs]), sum([G_diff_elem.mean() for G_diff_elem in G_diffs])/float(len(G)),
            max([G_diff_elem.max() for G_diff_elem in G_diffs]) )
