import sys
from pylearn2.utils import serial

model_path = sys.argv[1]
model = serial.load(model_path)
print model.monitor.examples_seen
