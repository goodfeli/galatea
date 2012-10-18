from matplotlib import pyplot
from pylearn2.utils import serial
import sys
import numpy as np

_, model_path = sys.argv

model = serial.load(model_path)

monitor = model.monitor

obj = monitor.channels['objective']

val_record = obj.val_record

val_record = np.asarray(val_record)
new = val_record[1:]
old = val_record[:-1]

decrease = old - new
prop_decrease = decrease / old

print 'min prop decrease: ',prop_decrease.min()

pyplot.plot(prop_decrease)
pyplot.show()
