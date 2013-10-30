import numpy as np

final_diffs = np.load('final_diffs.npy')

from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(final_diffs)
plt.show()
