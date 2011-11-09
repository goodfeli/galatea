#arg1: model to evaluate
#arg2: batch size
#arg3: # batches

import sys

model_path = sys.argv[1]

from pylearn2.utils import serial

print 'loading model...'
model = serial.load(model_path)
print 'done'

batch_size = int(sys.argv[2])
num_batches = int(sys.argv[3])

print 'defining em functional...'
import theano.tensor as T

V = T.matrix("V")
model.make_pseudoparams()
obs = model.e_step.mean_field(V)

from galatea.s3c.s3c import S3C

needed_stats = S3C.expected_log_prob_vhs_needed_stats()

from galatea.s3c.s3c import SufficientStatistics

stats = SufficientStatistics.from_observations( needed_stats = needed_stats, X = V, ** obs )
em_functional = model.em_functional( stats = stats, H = obs['H'], sigma0 = obs['sigma0'], Sigma1 = obs['Sigma1'])
assert len(em_functional.type.broadcastable) == 0

print 'compiling function...'
from theano import function

f = function([V], em_functional)
print 'done'

print 'loading dataset...'
dataset = serial.load('${STL10_PATCHES}')
print 'done'

total = 0.0
for i in xrange(num_batches):
    X = dataset.get_batch_design(batch_size)
    em = f(X)
    print (i, em)
    total += em

total /= num_batches

print 'ave em functional value: '+str(total)

