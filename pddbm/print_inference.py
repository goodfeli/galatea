import sys
from pylearn2.utils import serial

ignore, model_path = sys.argv

model = serial.load(model_path)

ip = model.inference_procedure

d = ip.new_coeff_lists

for key in d:
    v = d[key]
    print key,':',len(v)
    print '\t',v
