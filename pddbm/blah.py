#runs inference in a trained model
#modifies the inference procedure to have the "aggressive"
#and "repeat_h" flags enabled, reports the number of iterations
#to convergence on several batches
from pylearn2.utils import serial
import sys
from pylearn2.config import yaml_parse
import theano.tensor as T

ignore, model_path, other = sys.argv

model = serial.load(model_path)
model.make_pseudoparams()

other = serial.load(other)
model.inference_procedure = other.inference_procedure
model.inference_procedure.model = model
model.inference_procedure.s3c_e_step.model = model.s3c
model.inference_procedure.dbm_ip.model = model.dbm

dataset = yaml_parse.load(model.dataset_yaml_src)


Y = T.matrix()

has_labels = model.dbm.num_classes > 0

if not has_labels:
    Y = None
    Yv = None

#model.s3c.W.set_value(other.s3c.W.get_value())
model.s3c.B_driver.set_value(other.s3c.B_driver.get_value())

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

