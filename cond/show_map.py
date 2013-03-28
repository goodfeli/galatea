from galatea.cond.neighbs import cifar10neighbs
from galatea.cond.neighbs import multichannel_neibs2imgs

import numpy as np

m = 10

img = np.zeros((m,32,32,3),dtype='int64')

for i in xrange(32):
    img[:,i,:,0] += i
    img[:,:,i,1] += 32+i
for i in xrange(m):
    img[i,:,:,2] += 64+i

from pylearn2.utils.image import show

r = 6
c = 6
mc = 32-c+1
mr = 32-r+1

from theano.tensor import as_tensor_variable
neibs = cifar10neighbs(img,(r,c))
#neibs = neibs.reshape((m*mr*mr*3,r*c))
#from theano.sandbox.neighbours import neibs2images
#from theano.printing import Print
#neibs = Print('neibs',attrs=['shape'])(neibs)
#neibs = neibs.T.reshape((m*mr*mc*r*c*3,1))
#convmap = neibs2images(neibs, (1,1), (3*r*c,m,mr,mc))
#convmap = convmap.dimshuffle(1,2,3,0)
convmap = multichannel_neibs2imgs(neibs,m,mr,mc,3,r,c)
from theano import function
convmap = function([],convmap)()

print 'checking img'
for i in xrange(m):
    for j in xrange(32):
        for k in xrange(32):
            assert img[i,j,k,0] < 32
            assert img[i,j,k,1] >= 32
            assert img[i,j,k,1] < 64
            assert img[i,j,k,2] >= 64
print 'decoding img'
decoded_img = img - np.asarray([0,32,64])
decoded_img = np.cast['float32'](decoded_img)
decoded_img /= np.asarray([31.,31.,float(m)])

decoded_map = np.zeros((m,mr,mc,3))

ch = r * c * 3
assert convmap.shape[-1] == ch

for i in xrange(m):
    for j in xrange(mr):
        for k in xrange(mc):
            r = 0.
            g = 0.
            b = 0.
            r_count = 0
            g_count = 0
            b_count = 0

            for l in xrange(ch):
                elem = convmap[i,j,k,l]

                if elem < 32:
                    r_count += 1
                    r += float(elem)/31.
                if elem >= 32 and elem < 64:
                    g_count += 1
                    g += float(elem-32)/31.
                if elem >= 64:
                    b_count += 1
                    b += float(elem-64)/float(m)
            decoded_map[i,j,k,:] = np.asarray( [ r/(1e-12+float(r_count)),
                                                 g/(1e-12+float(g_count)),
                                                 b/(1e-12+float(b_count))])

for  i in xrange(m):
    show(decoded_img[i,:,:,:])
    show(decoded_map[i,:,:,:])

    x = raw_input()
