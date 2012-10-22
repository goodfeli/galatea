from matplotlib import pyplot
from pylearn2.utils import serial
import sys
import numpy as np

_, model_path = sys.argv

model = serial.load(model_path)

monitor = model.monitor

obj = monitor.channels['valid_err']

val_record = obj.val_record

running_min = [ val_record[0] ]
for elem in val_record[1:]:
    running_min.append(min(running_min[-1], elem))

val_record = np.asarray(val_record)
running_min = np.asarray(running_min)


new = val_record[1:]
old = running_min[:-1]

decrease = old - new
prop_decrease = decrease / old

print 'min prop decrease: ',prop_decrease.min()

pyplot.plot(prop_decrease)
pyplot.hold(True)
pyplot.plot([.01]*prop_decrease.shape[0])
pyplot.show()
