from gui import PatchViewer
from util import serial
import sys
import numpy as N
import numpy.linalg as NL

mu = serial.load(sys.argv[1])

if len(sys.argv) > 2:
    w = serial.load(sys.argv[2])

    mu = N.dot(mu,NL.inv(w[1]))

    filter = w[2][0]
    mu_input = mu
    mu = N.zeros((mu.shape[0],mu.shape[1]+filter.shape[0]))
    read_pos = 0
    for i in xrange(mu.shape[1]):
        if i in filter:
            continue
        else:
            mu[:,i] = mu_input[:,read_pos]
            read_pos += 1            


        

    mu += w[0]

[k,n] = mu.shape

print str(k)+' means survived'

factors = [ ]
o = k
f = 2
while f <= o:
    if (o/f)*f == o:
        factors.append(f)
        o = o/f
    else:
        f+= 1

rows = 1
cols = 1
for i in xrange(0,len(factors),2):
    rows *= factors[i]
for i in xrange(1,len(factors),2):
    cols *= factors[i]

if rows > cols:
    temp = rows
    rows = cols
    cols = temp

pv = PatchViewer.PatchViewer((rows,cols),(28,28))

for i in xrange(k):
    pv.add_patch( (mu[i,:].reshape(28,28)-.5)*2.0 )

pv.show()

print 'waiting...'
x = raw_input()
print '...running'

