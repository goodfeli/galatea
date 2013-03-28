rows = 10
cols = 10

from pylearn2.models.dbm import load_matlab_dbm
model = load_matlab_dbm('after_joint_train.mat', num_chains = rows * cols)
#model = load_matlab_dbm('after_backprop.mat', num_chains = rows * cols)
dbm = model

from theano import function
import theano.tensor as T

sample_func = function([],updates = model.get_sampling_updates())

render_func = function([],T.nnet.sigmoid(T.dot(dbm.H_chains[0],dbm.W[0].T)+dbm.bias_vis))

from pylearn2.datasets.mnist import MNIST

dataset = MNIST(which_set = 'train')
X = dataset.get_batch_design(rows*cols)

model.V_chains.set_value(X)



for i in xrange(200):
    print i
    sample_func()

from pylearn2.gui.patch_viewer import make_viewer



pv = make_viewer(dataset.adjust_for_viewer(render_func()))

pv.show()
