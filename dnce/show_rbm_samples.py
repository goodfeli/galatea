import sys
import theano.sandbox.rng_mrg
RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import theano.tensor as T
from theano import function
from pylearn2.gui.patch_viewer import make_viewer

ignore, model_path = sys.argv

model = serial.load(model_path)
dataset = yaml_parse.load(model.dataset_yaml_src)

theano_rng = RandomStreams(42)

X = T.matrix()
Y, ignore = model.gibbs_step_for_v(X, theano_rng)
print type(Y)

f = function([X],Y)

X = dataset.get_batch_design(100)

while True:
    V = dataset.adjust_for_viewer(X)
    viewer = make_viewer(V, is_color = X.shape[1] % 3 == 0)
    viewer.show()

    print 'Waiting...'
    x = raw_input()
    if x == 'q':
        break
    print 'Running...'

    num_updates = 1

    try:
        num_updates = int(x)
    except:
        pass

    for i in xrange(num_updates):
        X = f(X)

