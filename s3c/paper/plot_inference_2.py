import theano.tensor as T
from pylearn2.utils import serial

model_path = 'rpla_p5_interm.pkl'


model = serial.load(model_path)

#model.make_Bwp()
model.make_pseudoparams()

stl10 = model.dataset_yaml_src.find('stl10') != -1

if not stl10:
    raise NotImplementedError("Doesn't support CIFAR10 yet")

if stl10:
    dataset = serial.load("${PYLEARN2_DATA_PATH}/stl10/stl10_patches/data.pkl")

V_var = T.matrix()

#history = model.e_step.mean_field(V = V_var, return_history = True)
history = model.e_step.variational_inference(V = V_var, return_history = True)

feature_type = 'exp_h'

if feature_type == 'exp_h':
    outputs = [ hist_elem['H_hat'].mean() for hist_elem in history ]
else:
    raise NotImplementedError()

from theano import function
f = function([V_var], outputs= outputs)

import matplotlib.pyplot as plt

V = dataset.get_batch_design(1)
y = f(V)

plt.plot(y,'.')

ax = plt.gca()


plt.title('Sparsification during inference',fontsize=20)
plt.xlabel('Damped parallel fixed point updates',fontsize=18)
plt.ylabel('Mean of $\mathbb{E}_Q [h]$',fontsize=18)

plt.show()

