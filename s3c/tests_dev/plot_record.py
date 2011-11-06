import numpy as np

x = np.load('x.npy')
record = np.load('record.npy')
analytical = np.zeros(record.shape)+np.load('analytical.npy')

from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,record)
ax.plot(x,analytical)
ax.set_ylim(analytical[0]-3.*record.std(),analytical[0]+3.*record.std())
plt.show()

print 'final error: ',np.abs(analytical[-1]-record[-1])
