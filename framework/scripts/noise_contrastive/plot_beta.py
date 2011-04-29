from framework.utils import serial
import sys

model = serial.load(sys.argv[1])

import matplotlib.pyplot as plt

xs = []
ys  = []

for t in model.error_record:
    xs.append(t[0])
    ys.append(t[3])

plt.plot(xs,ys)
plt.show()
