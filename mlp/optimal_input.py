import sys

_, model_path, layer_idx  = sys.argv[0:3]
idxs = sys.argv[3:]
# note: idxs must include the batch

layer_idx = int(layer_idx)
idxs = [int(idx) for idx in idxs]

from pylearn2.utils import serial
model = serial.load(model_path)

input_space = model.get_input_space()

from pylearn2.utils import sharedX
X = sharedX(input_space.get_origin_batch(1))

from theano import tensor as T
normed = X / (1e-7 + T.sqrt(1e-7 + T.sqr(X).sum()))

outputs = model.fprop(normed, return_all=True)

output = outputs[layer_idx]
neuron = output[tuple(idxs)]

from pylearn2.optimization.batch_gradient_descent import BatchGradientDescent

bgd = BatchGradientDescent(objective=-neuron,
        params=[X],
        inputs=None,
        max_iter=100,
        lr_scalers=None,
        verbose=3,
        tol=None,
        init_alpha=None,
        min_init_alpha=1e-3,
        reset_alpha=True,
        conjugate=True,
        gradients=None,
        gradient_updates=None,
        accumulate=False,
        theano_function_mode=None,
        param_constrainers=None)

bgd.minimize()


X = normed.eval()[:,:,:,0].transpose(1,2,0)
import numpy as np
X /= np.abs(X).max()
print (X.min(), X.max())

from pylearn2.utils.image import show
show(X)
