import sys
from pylearn2.utils.string_utils import number_aware_alphabetical_key

ignore, model_path = sys.argv

from pylearn2.utils import serial

model = serial.load(model_path)

keys = model.monitor.channels.keys()

kls = [ key for key in keys if key.startswith('trunc_KL_') ]

kls.sort(key = number_aware_alphabetical_key)


worst_uphills = {}

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
            diff = cur_val - prev_val
            if diff > 1e-4:
                if cur_key not in worst_uphills or worst_uphills[cur_key] < cur_val:
                    worst_uphills[cur_key] = diff

        if cur_val == prev_val:
            print cur_key,' made no progress on step ',str(i)

print "\n\n\nWorst uphills at each step:"
for cur_key in sorted(worst_uphills.keys() , key = number_aware_alphabetical_key):
    print '\t',cur_key,':',worst_uphills[cur_key]
