import sys
import theano.tensor as T
from pylearn2.utils import serial

model_path = 'rpla_p5_interm.pkl'

shuffle = False
if len(sys.argv) > 2:
    assert sys.argv[2] == '--shuffle'
    #shuffle option gets random examples by skipping model.nhid ahead at the start
    #this way if using a random patches model we don't see the random patches that the
    #model uses as weights
    shuffle = True

model = serial.load(model_path)

#model.make_Bwp()
model.make_pseudoparams()


model.bias_hid.set_value(model.bias_hid.get_value() * 1.3)

stl10 = model.dataset_yaml_src.find('stl10') != -1

if not stl10:
    raise NotImplementedError("Doesn't support CIFAR10 yet")

if stl10:
    dataset = serial.load("${PYLEARN2_DATA_PATH}/stl10/stl10_patches/data.pkl")

if shuffle:
    dataset.get_batch_design(model.nhid)

V_var = T.matrix()

mean_field = model.e_step.variational_inference(V = V_var)

feature_type = 'exp_h'

if feature_type == 'exp_h':
    outputs = mean_field['H_hat']
else:
    raise NotImplementedError()

from theano import function
f = function([V_var], outputs= outputs)

import matplotlib.pyplot as plt

V = dataset.get_batch_design(100)
y = f(V)

y = y.reshape( y.shape[0] * y.shape[1])


plt.hist(y, bins = 20, log = True)

#ax = plt.gca()
#ax.set_yscale('symlog' )

plt.title('Distribution of $\mathbb{E}[h_i] $',fontsize=20)
plt.xlabel('$ \mathbb{E}[h_i] $', fontsize=18)
plt.ylabel('log number of occurrences',fontsize=18)

plt.show()


print (y < .01).mean()
