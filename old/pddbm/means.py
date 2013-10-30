import sys
import numpy as np

X = np.load(sys.argv[1])

print X.mean()
print 'the above seems to have very bad numerical accuracy'
print X.mean(axis=0).mean()

m = X.mean(axis=0)

print (m.min(),m.mean(),m.max())


n = 0.
mean = 0.

for r in xrange(X.shape[0]):
    for c in xrange(X.shape[1]):
        n = n + 1
        delta = X[r,c] - mean
        mean = mean + delta / n

print mean


