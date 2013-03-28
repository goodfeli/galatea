import numpy as np

x = np.load('x.npy')
diffs = np.load('diffs.npy')

from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,diffs)
ax.set_ylim(0,diffs[-1]*2.)
plt.show()
