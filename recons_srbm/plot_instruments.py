import numpy as N
import sys
from framework.utils import serial
import matplotlib.pyplot as plt

model = serial.load(sys.argv[1])

d = model.instrument_record.d

keys = []

for key in d.keys():
    if type(key) == type(''):
        keys.append(key)
    elif type(key) == type((1,)):
        assert len(key) == 2
        assert type(key[0]) == type('')
        assert type(key[1]) == type(1)
        final_key = key[0]+'*'
        if final_key not in keys:
            keys.append(final_key)

keys = sorted(keys)

while True:
    for i, key in enumerate(keys):
        print str(i)+'. '+key
    print 'q. Quit'

    x = raw_input()

    if x == 'q':
        quit()
    c = int(x)

    key = keys[c]

    if key[-1] == '*':
        key = key[:-1]
        print 'choose entry (len is '+str(len(d['examples_seen']))+') '
        x = raw_input()
        c = int(x)

        xs = []
        ys = []

        i = 1
        while (key,i) in d:
            xs.append(i)
            ys.append(d[(key,i)][c])
            i += 1
        #
        print ys
        plt.plot(xs,ys)
        plt.show()
    else:
        plt.plot(N.asarray(d['examples_seen']),N.asarray(d[key]))
        plt.show()
    #
#
