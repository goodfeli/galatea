import sys
_, model_path = sys.argv
from pylearn2.utils import serial
model = serial.load(model_path)
d = model.discriminator
import gc
del model
gc.collect()
from pylearn2.utils import sharedX
X = sharedX(d.get_input_space().get_origin_batch(1))
obj =  -d.fprop(X).sum()
from pylearn2.optimization.batch_gradient_descent import BatchGradientDescent as BGD
import theano.tensor as T
def norm_constraint(updates):
    assert X in updates
    updates[X] = updates[X] / (1e-7 + T.sqrt(T.sqr(X).sum()))
opt = BGD(objective=obj, params=[X], param_constrainers=[norm_constraint], conjugate=True, reset_conjugate=False,
        reset_alpha=False, line_search_mode='exhaustive', verbose=3, max_iter=20)
results = []
import numpy as np
rng = np.random.RandomState([1, 2, 3])
for i in xrange(10):
    X.set_value(rng.randn(*X.get_value().shape).astype(X.dtype) / 10.)
    opt.minimize()
    Xv = X.dimshuffle(3, 1, 2, 0).eval()
    results.append(Xv)
X = np.concatenate(results, axis=0)
from pylearn2.gui.patch_viewer import make_viewer
v = make_viewer(X)
v.show()

