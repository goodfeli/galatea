
diffs = []

import numpy as np
import sys
path = sys.argv[1]

X_src = np.load(path)


for n in xrange(1,101):
    print n

    X = X_src[0:n*X_src.shape[0]/100,:]

    print 'computing'
    m1 = X.mean(dtype='float64')
    m2 = X.mean(axis=0,dtype='float64').mean()
    diff = abs(m1-m2)
    print diff
    diffs.append(diff)

from matplotlib import pyplot as plt
plt.plot(diffs)
