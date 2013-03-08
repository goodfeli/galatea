import sys
_, model_path = sys.argv

from pylearn2.utils import serial
model = serial.load(model_path)

monitor = model.monitor
channels = monitor.channels
train_y_misclass = channels['train_y_misclass']
train_y_misclass = train_y_misclass.val_record
mn = min(train_y_misclass)
print mn
