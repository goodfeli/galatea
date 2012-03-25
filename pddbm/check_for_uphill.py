import sys

ignore, model_path = sys.argv

from pylearn2.utils import serial

model = serial.load(model_path)

keys = model.monitor.channels.keys()

kls = [ key for key in keys if key.startswith('trunc_KL_') ]

from pylearn2.utils.string_utils import number_aware_alphabetical_key

kls.sort(key = number_aware_alphabetical_key)

for i in xrange(len(model.monitor.channels[kls[0]].val_record)):
    for j in xrange(1,len(kls)):
        cur_key = kls[j]
        prev_key = kls[j-1]

        cur_channel = model.monitor.channels[cur_key]
        prev_channel = model.monitor.channels[prev_key]

        cur_val = cur_channel.val_record[i]
        prev_val = prev_channel.val_record[i]

        if cur_val > prev_val:
            print cur_key,' went uphill by '+str(cur_val - prev_val)+' on step ',str(i)

        if cur_val == prev_val:
            print cur_key,' made no progress on step ',str(i)
