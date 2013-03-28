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
#model.set_dtype('float32')
if hasattr(model,'make_pseudoparams'):
    model.make_pseudoparams()

dataset = yaml_parse.load(model.dataset_yaml_src)

Xv , Yv = dataset.get_batch_design(100, include_labels = True)

X = T.matrix()
Y = T.matrix()

try:
    has_labels = model.dbm.num_classes > 0
except:
    try:
        has_labels = model.num_classes > 0
    except:
        has_labels = False

if not has_labels:
    Y = None
    Yv = None

if hasattr(model,'e_step'):
    model.inference_procedure = model.e_step

python = not hasattr(model.inference_procedure,'infer')
print type(model.inference_procedure)
print dir(model.inference_procedure)
if hasattr(model.inference_procedure,'layer_schedule') and model.inference_procedure.layer_schedule is None:
    model.inference_procedure.layer_schedule = [0,1] * 10
obs_hist = model.inference_procedure.infer(X,Y = Y,return_history = True)

mean_acts = []

which_var = var
for obs in obs_hist:
    var = obs[which_var]
    if idx is not None:
        assert isinstance(var,list) or isinstance(var,tuple)
        var = var[idx]
    mean_acts.append(var.mean())

inputs = [ X ]
input_vals = [ Xv ]
if has_labels:
    inputs.append(Y)
    input_vals.append(Yv)

f = function(inputs, mean_acts, on_unused_input = 'ignore')


mean_acts = f(*input_vals)

plt.plot(mean_acts)
plt.show()
