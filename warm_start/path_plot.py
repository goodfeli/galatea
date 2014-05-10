import sys

_, model_path = sys.argv

from pylearn2.utils import serial

model = serial.load(model_path)

monitor = model.monitor
channels = monitor.channels

soln_coord = channels['soln_coord']
residual_norm = channels['residual_norm']

from matplotlib import pyplot

pyplot.plot(soln_coord.val_record, residual_norm.val_record,
        marker = 'x')

if 'sheep_coord' in channels:
    pyplot.hold(True)
    pyplot.plot(soln_coord.val_record, channels['sheep_coord'].val_record,
            marker = 'o')

pyplot.show()
