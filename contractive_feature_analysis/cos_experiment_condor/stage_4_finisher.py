job_name = 'cfa_cos_tanh'

from pylearn2.utils import serial
import SkyNet
import numpy as N
from scipy.linalg import eigh
import time

SkyNet.set_job_name(job_name)
components = SkyNet.get_dir_path('components')

num_examples = serial.load(components+'/num_examples.pkl')
chunk_size = serial.load(components+'/chunk_size.pkl')
batch_size = serial.load(components+'/batch_size.pkl')
expanded_dim = serial.load(components+'/expanded_dim.pkl')
whitened_dim = serial.load(components+'/whitener.pkl').get_weights().shape[1]

instability_matrices = SkyNet.get_dir_path('instability_matrices')
components = SkyNet.get_dir_path('components')

assert num_examples % chunk_size == 0
num_chunks = num_examples / chunk_size

G = N.zeros((whitened_dim, whitened_dim) )

print 'Summing up instability matrices'
for b in xrange(0,num_examples,chunk_size):
    tmp = N.load(instability_matrices+'/instability_matrix_%d.npy' % b)
    G += tmp


print 'Finding eigenvectors'
t1 = time.time()
v, W = eigh(G)
t2 = time.time()
print (t2-t1),' seconds'

serial.save(components+'/v.pkl',v)
serial.save(components+'/W.pkl',W)
