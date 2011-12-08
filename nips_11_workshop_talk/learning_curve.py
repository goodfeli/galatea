from pylearn2.utils import serial

model = serial.load('G3.pkl')

ch = model.monitor.channels['em_functional']

import matplotlib.pyplot as plt

plt.plot(ch.example_record, ch.val_record)

plt.title('Learning over time')
plt.xlabel('Examples seen (1 update per 100 examples)')
plt.ylabel('Energy functional on heldout set')

plt.show()
