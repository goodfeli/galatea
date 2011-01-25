#!/usr/bin/python
import sys

import pylearn.datasets.utlc as utlc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

print "Usage: to_hist.py {avicenna, ule, harry, rita, sylvester, terry}"

assert len(sys.argv)==2
assert sys.argv[1] in ['avicenna','ule','harry','rita','sylvester','terry']
name = sys.argv[1]

tops=dict(avicenna=0.03,ule=0.01,harry=0.001)
bins=dict(harry=30)
if name in tops:
    top = tops[name]
else:
    top = 0.01
if name in bins:
    bin = bins[name]
else:
    bin = 150
if name != "terry":
    train, valid, test = utlc.load_ndarray_dataset(name, normalize=False)
    train = train.flatten()[0:1e6]
    print "For dataset %s their is %f%% of number that are 0"%(name, float((train==0).sum())/train.size)
    print "min=%f, max=%f, mean=%f, std=%f of the train part of this dataset"%(
    train.min(), train.max(), train.mean(), train.std())
else:
    train, valid, test = utlc.load_sparse_dataset(name, normalize=False)
    train = train.data[0:1e6]
del valid
del test
x = train
print x.size
fig = plt.figure()
ax = fig.add_subplot(111)

# the histogram of the data
n, bins, patches = ax.hist(x, bin, normed=1, facecolor='green', alpha=0.75)

ax.set_xlabel('Value the train of this dataset.')
ax.set_ylabel('Probability')
ax.set_title("Histogramme for dataset "+name)
ax.set_xlim(train.min(), train.max())
ax.set_ylim(0, top)
ax.grid(True)
    
plt.show()
