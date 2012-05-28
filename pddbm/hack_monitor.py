#turns kl_fail_log into a monitoring channel

import sys
from pylearn2.utils import serial
from pylearn2.monitor import MonitorChannel

ignore, model_path = sys.argv

model = serial.load(model_path)

name = 'kl_fail'
if hasattr(model,'s3c'):
    s3c = model.s3c
else:
    s3c = model
channel = MonitorChannel(None,s3c.W.sum(),name)
model.monitor.channels[name] = channel

for ex, val in model.kl_fail_log:
    channel.example_record.append(ex)
    channel.val_record.append(val)

serial.save(model_path, model)

