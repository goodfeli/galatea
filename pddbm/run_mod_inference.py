#runs inference in a trained model
#modifies the inference procedure to have the "aggressive"
#and "repeat_h" flags enabled, reports the number of iterations
#to convergence on several batches
from pylearn2.utils import serial
import sys
from pylearn2.config import yaml_parse
import theano.tensor as T

ignore, model_path = sys.argv

model = serial.load(model_path)
model.make_pseudoparams()

dataset = yaml_parse.load(model.dataset_yaml_src)


Y = T.matrix()

has_labels = model.dbm.num_classes > 0

if not has_labels:
    Y = None
    Yv = None

python = not hasattr(model.inference_procedure,'infer')
assert python

model.inference_procedure.aggressive = True
model.inference_procedure.repeat_h = False
model.inference_procedure.s_cg_steps = 7
model.inference_procedure.redo_theano()

channels = model.monitor.channels
if 'niter' in channels:
    print 'old niter: '
    print channels['niter'].val_record[-1]

while True:
    Xv , Yv = dataset.get_batch_design(100, include_labels = True)
    print 'running inference'
    model.inference_procedure.update_var_params( Xv )
    print model.inference_procedure.niter.get_value()
    print model.inference_procedure.time.get_value()

