from pylearn2.utils import serial
import sys
from pylearn2.config.yaml_parse import load
import theano.tensor as T
from theano import function
import numpy as np
import matplotlib.pyplot as plt
from theano import config
#float32 seems to break David's code
config.floatX = 'float64'

print 'loading model'
model_path = '/u/goodfeli/galatea/s3c/config/stl/rpla_penalized_5_deploy.pkl'
model = serial.load(model_path)
model.make_pseudoparams()
model.set_dtype(config.floatX)

print 'loading dataset'
dataset = load(model.dataset_yaml_src)

print 'compiling function'
V = T.fmatrix()
obs = model.e_step.variational_inference(V)
H = obs['H_hat']
S = obs['S_hat']
HS = abs(H*S)

f = function([V], HS)

batch_size = 5000


num_trajectories = 5

mu_scale_schedule = [ 10., 6.666, 3.333, 0., 3.3333, 6.666, 10. ]
p_scale_schedule = [  1.,  2.,    2.,    1., .5,     .5,    1.  ]

trajectories = []

for i in xrange(num_trajectories):
    trajectories.append( { 'x' : [], 'y' : [] } )

mu_orig = model.mu.get_value()
b = model.bias_hid.get_value()
p_orig = 1./(1.+np.exp(-b))



X = dataset.get_batch_design(batch_size)


for j in xrange(len(mu_scale_schedule)):
    print 'inference for trajectory point ',j

    mu = mu_orig * mu_scale_schedule[j]
    p = p_orig * p_scale_schedule[j]

    model.mu.set_value(mu)
    model.bias_hid.set_value( np.log( - p / (p - 1) ) )


    HS = f(X)

    act_prob = (HS[:,0:num_trajectories] > .01).mean(axis=0)

    act_mean = np.zeros(act_prob[0:num_trajectories].shape)
    for i in xrange(num_trajectories):
        s = HS[:,i]
        s = s[s > .01]
        act_mean[i] = s.mean()

        trajectories[i]['x'].append(act_prob[i])
        trajectories[i]['y'].append(act_mean[i])

print "drawing plot"
plt.hold(True)

for trajectory in trajectories:
    plt.plot(trajectory['x'],trajectory['y'])
    trajectory['x'] = []
    trajectory['y'] = []

plt.show()


alpha_schedule = [ .2/5000., .5/5000., 1./5000., 2./5000., 5./5000. ]


assert num_trajectories == len(trajectories)

for j, alpha in enumerate(alpha_schedule):
    print 'j = ',j,'; alpha = ',alpha
    from sklearn.decomposition import sparse_encode
    print 'running SC ',j
    HS = sparse_encode( model.W.get_value(), X.T, alpha = alpha, algorithm='lasso_cd').T
    assert HS.shape == (5000,1600)
    print 'done encoding'

    HS = np.abs(HS)


    if np.any(np.isnan(HS)):
        print 'has nans'

    if np.any(np.isinf(HS)):
        print 'has infs'

    print 'HS shape ',HS.shape
    print 'HS subtensor shape ',HS[0:num_trajectories].shape
    act_prob = (HS[:,0:num_trajectories] > .01).mean(axis=0)
    print act_prob.shape
    assert len(act_prob.shape) == 1

    act_mean = np.zeros(act_prob.shape)
    assert act_mean.shape[0] == num_trajectories
    for i in xrange(act_mean.shape[0]):
        s = HS[:,i]
        s = s[s > .01]
        act_mean[i] = s.mean()

        trajectories[i]['x'].append(act_prob[i])
        trajectories[i]['y'].append(act_mean[i])


print "drawing plot"
plt.hold(False)

for trajectory in trajectories:
    plt.plot(trajectory['x'],trajectory['y'])
    print (trajectory['x'],trajectory['y'])
    plt.hold(True)

plt.show()

