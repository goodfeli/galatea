import matplotlib.pyplot as plt
import numpy as np

b = 10.

y = np.random.uniform(np.log(.001)/np.log(b), np.log(.01)/np.log(b), (10000,))
x = b ** y

plt.hist(x, bins=100)

plt.show()
