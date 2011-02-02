"""
Load a sparse UTLC dataset and saves it in the coo_matrix sparse format.
"""

import os
import sys

import numpy
import cPickle
import scipy.sparse
import theano

import functools as fc

# temporaire
ROOT_PATH = '/part/02/Tmp/saintjaf/utlc/raw'

datasets_avail = ["terry", "ule"]
subset = ["devel", "valid", "final"]

datasets_shape = {
		"terry_devel" : 217034,
		"ule_devel" : 26808 }

# normalize functions
def normalize_gaussian(mean, std, set):
	return (set - mean) / std

def normalize_maximum(max, set):
	return set / max

# statistics
terry_max = 300.0
ule_max = 255.0

datasets_normalizer= {
		"terry"     : fc.partial(normalize_maximum, terry_max) ,
		"ule"       : fc.partial(normalize_maximum, ule_max) }

def load_dataset(name, subset, permute_train=True, normalize=True):
    if not os.path.exists(os.path.join(ROOT_PATH,name+'_text')):
        raise Exception("The directory with the original data does not exists")

    data = numpy.fromfile(os.path.join(ROOT_PATH,name+'_text',name+'_'+subset+'.data'),
                           dtype=numpy.float32, sep=' ')


	# fast and dirty.
    if name == "terry":
        data = data.reshape(data.shape[0]/3, 3)

        if normalize:
            data[:,2] = datasets_normalizer[name](data[:,2])

        if permute_train:
            size = 4096
            if subset == "devel":
                size = datasets_shape[name + "_devel"]

            rng = numpy.random.RandomState([1,2,3])
            perm = rng.permutation(size)

	    # tricky, data = data[perm] will in fact permute nothing on a sparse format
	    # as we move around the value with its coordinate.
	    # The call to coo_matrix will put the coordinate in the original order.
            for i in range(data.shape[0]):
		# The first row os the sparse marix is empty as the sparce
		# matrix index start at 0, but in the file it start at 1
                data[i,0] = perm[data[i,0]-1]
        data = scipy.sparse.coo_matrix((data[:,2], (data[:,0],data[:,1])))
	data = scipy.sparse.csr_matrix(data)
    else:

        size = 4096
        if subset == "devel":
            size = datasets_shape[name + "_devel"]
        data = data.reshape(size, data.shape[0]/size)

        if normalize:
            data = datasets_normalizer[name](data)

        if permute_train:
            rng = numpy.random.RandomState([1,2,3])
            perm = rng.permutation(data.shape[0])
            data = data[perm]

        data = scipy.sparse.csr_matrix(data)

    return data

def write_dataset(name, subset, dataset):
    print "Wrinting dataset", name
    cPickle.dump(dataset, open(name+'_'+subset+'.npy','wb'))

todo = sys.argv[1:]
if len(sys.argv)<=1 or any([d not in datasets_avail for d in todo]):
    print "Usage: to_npy.py [terry, ule]"
    quit(1)

print "Will process datasets:", todo
for data in todo:
	for s in subset:
		write_dataset(data, s, load_dataset(data,s))
