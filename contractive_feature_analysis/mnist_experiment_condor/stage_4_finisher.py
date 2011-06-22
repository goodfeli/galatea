job_name = 'cfa'

from pylearn2.utils import serial
import SkyNet
import numpy as N

SkyNet.set_job_name(job_name)
components = SkyNet.get_dir_path('components')

num_examples = serial.load(components+'/num_examples.pkl')
chunk_size = serial.load(components+'/chunk_size.pkl')
batch_size = serial.load(components+'/batch_size.pkl')
expaned_dim = serial.load(components+'/expanded_dim.pkl')

instability_matrices = SkyNet.get_dir_path('instability_matrices')

N.save('instability_matrix_%d.npy' % idx, G)

assert num_examples % chunk_size == 0
num_chunks = num_examples / chunk_size

G = N.zeros((expanded_dim, expanded_dim) )

for b in xrange(0,num_examples,chunk_size):
    if b != 0:
        command+=','

    tmp = N.load(instability_matrices+'/instability_matrix_%d.npy' % b)

    command += str(b)
command += '}}"'

SkyNet.launch_job(command)

print 'Finding eigenvectors'
t1 = time.time()
v, W = eigh(G)
t2 = time.time()
print (t2-t1),' seconds'

results = {}
results['v'] = v
results['W'] = W
results['whitener'] = whitener
results['pca_model'] = pca_model

serial.save('mnist_results.pkl',results)
