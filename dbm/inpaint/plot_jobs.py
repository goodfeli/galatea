import sys
from pylearn2.utils import serial
channel_name = 'valid_err'

for path in sys.argv[1:]:
    print path
    try:
        model = serial.load(path, retry = False)
    except:
        continue
    monitor = model.monitor
    channel = monitor.channels[channel_name]
    print channel.val_record[-1]
