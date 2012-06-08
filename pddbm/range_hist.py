from pylearn2.utils import serial
import sys
from pylearn2.config import yaml_parse
from theano import function
import theano.tensor as T
import matplotlib.pyplot as plt

if len(sys.argv) == 3:
    ignore, model_path, var = sys.argv
    idx = None
else:
    ignore, model_path, var, idx = sys.argv
    idx = int(idx)

model = serial.load(model_path)
model.make_pseudoparams()

dataset = yaml_parse.load(model.dataset_yaml_src)

Xv , Yv = dataset.get_batch_design(100, include_labels = True)

X = T.matrix()
Y = T.matrix()

has_labels = model.dbm.num_classes > 0

if not has_labels:
    Y = None
    Yv = None

ip = model.inference_procedure
python = not hasattr(ip,'infer')
if python:
    ip.redo_theano()
    obs = ip.hidden_obs
else:
    obs = ip.infer(X,Y)

var = obs[var]
if idx is not None:
    assert isinstance(var,list) or isinstance(var,tuple)
    var = var[idx]

inputs = [ X ]
input_vals = [ Xv ]
if has_labels:
    inputs.append(Y)
    input_vals.append(Yv)

f = function(inputs, var, on_unused_input = 'ignore')

if python:
    ip.update_var_params(*input_vals)
var_val = f(*input_vals)

maxes = var_val.max(axis=0)
mins = var_val.min(axis=0)
ranges = maxes - mins

plt.hist(ranges, bins = int(ranges.shape[0] / 10))
plt.show()
