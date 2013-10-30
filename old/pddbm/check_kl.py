#turns kl_fail_log into a monitoring channel

import sys
from pylearn2.utils import serial

ignore, model_path = sys.argv

model = serial.load(model_path)


for ex, val in model.kl_fail_log:
    assert val < .05

print 'success'
