from pylearn2.utils import serial
import sys
from pylearn2.config import yaml_parse
import numpy as np
from pylearn2.gui import patch_viewer

patch_rescale = True

model_path = sys.argv[1]

model = serial.load(model_path)



from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
theano_rng = RandomStreams(42)
assert hasattr(model.dbm,'V_chains') and model.dbm.V_chains is not None

#print model.dbm.V_chains.get_value()

design_examples_var = model.s3c.random_design_matrix(batch_size = model.dbm.negative_chains,
        theano_rng = theano_rng, H_sample = model.dbm.V_chains,
        S_sample = model.s3c.mu.dimshuffle('x',0), full_sample = False)
from theano import function
print 'compiling sampling function'
f = function([],design_examples_var)
print 'sampling'
design_examples = f()
print 'loading dataset'
dataset = yaml_parse.load(model.dataset_yaml_src)

Xm = design_examples
Xd = dataset.X

m_mn = Xm.mean(axis=0)
d_mn = Xd.mean(axis=0)

err = np.abs(m_mn - d_mn)

print 'data marginal means: ',(d_mn.min(),d_mn.mean(),d_mn.max())
print 'error in marginal mean: ',(err.min(),err.mean(),err.max())

#to get the variance right we need to include the effect of sampling with B
design_examples_var = model.s3c.random_design_matrix(batch_size = model.dbm.negative_chains,
        theano_rng = theano_rng, H_sample = model.dbm.V_chains, full_sample = True)
from theano import function
print 'compiling sampling function'
f = function([],design_examples_var)
print 'sampling'
Xm = f()

m_var = Xm.var(axis=0)
print 'computed model variance'
d_var = Xd.var(axis=0)
print 'computed data variance'

ratio = m_var / d_var
print 'computed ratio'

print 'model variance / data variance: ',(ratio.min(),ratio.mean(),ratio.max())

z_score = err / np.sqrt(d_var)
print 'error in marginal mean, expressed in terms of data standard deviations: ',(z_score.min(),z_score.mean(),z_score.max())



from matplotlib import pyplot

pyplot.scatter(err, ratio)

pyplot.xlabel('error in marginal mean')
pyplot.ylabel('model variance / data variance')

pyplot.show()
