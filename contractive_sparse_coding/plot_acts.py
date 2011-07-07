import sys
import matplotlib.pyplot as plt
from framework.utils import serial
import numpy as N

d = serial.load(sys.argv[1])

dots = N.load(d['dots'])
acts = N.load(d['acts'])
bvec = d['b']
gvec = d['g']

assert dots.shape == acts.shape

print 'overall ave act '+str(acts.mean())
means = acts.mean(axis=0)
print (means.min(),means.max())

plt.hist(means, bins=1000)
plt.show()

print 'waiting'
x = raw_input()
if x == 'q':
    quit()
print 'running'

for i in xrange(dots.shape[1]):
    b = bvec[i]
    g = gvec[i]

    plt.hold(False)
    a = acts[:,i]
    #plt.hexbin(dots[:,i],a)
    plt.scatter(dots[:,i],a)

    print 'expected act: '+str(a.mean())

    mn = dots[:,i].min()
    width = dots[:,i].max() - mn

    n = 1000
    xs = [ mn + width * float(i)/float(n-1) for i in xrange(n) ]

    ys = [ g/(1.+N.exp( -(x+b) )) for x in xs]

    plt.hold(True)
    plt.plot(xs,ys,'r')

    plt.show()
    print 'waiting'
    x = raw_input()
    if x == 'q':
        break
    print 'running'

    plt.hold(False)

    plt.hist(a, bins=20)

    plt.show()

    print 'waiting'
    x = raw_input()
    if x == 'q':
        break
    print 'running'
