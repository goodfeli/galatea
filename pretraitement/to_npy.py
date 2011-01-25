""" 
This is a script to transform the data from Unsupervised and Transfer Learning Challenge(UTLC)

see the README file for defail of what have been done.

dataset:
wget http://www.causality.inf.ethz.ch/ul_data/avicenna_text.zip
wget http://www.causality.inf.ethz.ch/ul_data/harry_text.zip
wget http://www.causality.inf.ethz.ch/ul_data/rita_text.zip
wget http://www.causality.inf.ethz.ch/ul_data/sylvester_text.zip
wget http://www.causality.inf.ethz.ch/ul_data/terry_text.zip
wget http://www.causality.inf.ethz.ch/ul_data/ule_text.zip

TODO: pylearn.io.sparse? That will use numpy.tofile?
TODO: pylearn.datasets/utlc.py to load all of them.
"""

import cPickle
import os
import sys

import numpy
import pylearn.io.filetensor
import scipy.sparse

ROOT_PATH = '/data/lisa/data/UTLC/'

datasets_avail = ["--avicenna","--harry","--rita","--sylvester","--terry","--ule"]
todo = sys.argv[1:]
if len(sys.argv)<=1 or any([d not in datasets_avail for d in todo]):
    print "Usage: to_npy.py {--avicenna,--harry,--rita,--sylvester,--terry,--ule}"

print "Will process datasets:", todo
def load_dataset(name, dtype=None, permute_train=False):
    """ 
    This version use much more memory
    as numpy.loadtxt use much more memory!
    
    But we don't loose the shape info in the file!
    """
    if not os.path.exists(os.path.join(ROOT_PATH,name+'_text')):
        raise Exception("The directory with the original data for %s is not their"%name)
    train = numpy.loadtxt(os.path.join(ROOT_PATH,name+'_text',name+'_devel.data'),
                          dtype=dtype)
    if permute_train:
        rng = numpy.random.RandomState([1,2,3])
        perm = rng.permutation(train.shape[0])
        train = train[perm]
    valid = numpy.loadtxt(os.path.join(ROOT_PATH,name+'_text',
                                       name+'_valid.data'),
                          dtype=dtype)
    test = numpy.loadtxt(os.path.join(ROOT_PATH,name+'_text',
                                      name+'_final.data'),
                         dtype=dtype)
    return train, valid, test

def load_dataset2(name, dtype=None, rows_size=None, permute_train=False):
    if not os.path.exists(os.path.join(ROOT_PATH,name+'_text')):
        raise Exception("The directory with the original data for %s is not their"%name)
    valid = numpy.fromfile(os.path.join(ROOT_PATH,name+'_text',
                                       name+'_valid.data'),
                          dtype=dtype, sep=' ')
    test = numpy.fromfile(os.path.join(ROOT_PATH,name+'_text',
                                      name+'_final.data'),
                         dtype=dtype, sep=' ')

    train = numpy.fromfile(os.path.join(ROOT_PATH,name+'_text',name+'_devel.data'),
                           dtype=dtype, sep=' ')
    if rows_size is not None:
        train = train.reshape(train.size/rows_size, rows_size)
        valid = valid.reshape(valid.size/rows_size, rows_size)
        test = test.reshape(test.size/rows_size, rows_size)
    if permute_train:
        assert rows_size is not None, "we need to know the number of row to permute the data!"
        rng = numpy.random.RandomState([1,2,3])
        perm = rng.permutation(train.shape[0])
        train = train[perm]
    return train, valid, test

def load_coo_matrix(name, dtype=None, rows_size=None, permute_train=False):
    if rows_size is None:
        train, valid, test = load_dataset(name, dtype, permute_train=False)
    else:
        train, valid, test = load_dataset2(name, dtype, rows_size=rows_size,
                                           permute_train=False)
    if permute_train:
        rng = numpy.random.RandomState([1,2,3])
        perm = rng.permutation(train.shape[0])
        train = train[perm]
    valid = scipy.sparse.coo_matrix((valid[:,2],(valid[:,0],valid[:,1])))
    test = scipy.sparse.coo_matrix((test[:,2],(test[:,0],test[:,1])))
    train = scipy.sparse.coo_matrix((train[:,2],(train[:,0],train[:,1])))

    return train, valid, test

def write_dataset(name, train, valid, test, pickle=False):
    print "Wrinting dataset", name
    if pickle:
        cPickle.dump(train, open(name+'_train.npy','wb'))
        cPickle.dump(valid, open(name+'_valid.npy','wb'))
        cPickle.dump(test, open(name+'_test.npy','wb'))    
    else:
        pylearn.io.filetensor.write(open(name+'_train.ft','wb'),train)
        pylearn.io.filetensor.write(open(name+'_valid.ft','wb'),valid)
        pylearn.io.filetensor.write(open(name+'_test.ft','wb'),test)


if "--avicenna" in todo:
    train, valid, test = load_dataset('avicenna', dtype='int16', permute_train=True)
    write_dataset('avicenna', train, valid, test)
    print "You must manually compress the created file with gzip!"


if "--harry" in todo:
#harry_train.ft 665M
#harry_train.ft.gz 17M
#dump of scipy.sparse.csr_matrix: 154M gzip: 20M
    train, valid, test = load_dataset2('harry', dtype='int16',
                                       rows_size=5000, permute_train=True)
    write_dataset('harry', train, valid, test)
    train = scipy.sparse.csr_matrix(train)
    valid = scipy.sparse.csr_matrix(valid)
    test = scipy.sparse.csr_matrix(test)
    write_dataset('harry', train, valid, test, pickle=True)
    print "You must manually compress the created file with gzip!"

if "--rita" in todo:
    train, valid, test = load_dataset2('rita', dtype='uint8',
                                       rows_size=7200, permute_train=True)
    write_dataset('rita', train, valid, test)
    print "You MUST NOT manually compress the created file with gzip! They are not smaller."

if "--sylvester" in todo:
#DONE, file not gzipped as it is not worth the size saved!~25% saved
    train, valid, test = load_dataset2('sylvester', dtype='int16',
                                       rows_size=100, permute_train=True)
    write_dataset('sylvester', train, valid, test)
    print "You must manually compress the created file with gzip!"

if "--terry" in todo:
#terry_train.npy 605M
    #must be int32 for the indexes
    train, valid, test = load_coo_matrix('terry', dtype='int32', 
                                         rows_size=3, permute_train=True)
    train = scipy.sparse.csr_matrix(train, dtype="int16")
    valid = scipy.sparse.csr_matrix(valid, dtype="int16")
    test = scipy.sparse.csr_matrix(test, dtype="int16")
    write_dataset('terry', train, valid, test, pickle=True)
    print "You must manually compress the created file with gzip!"

if "--ule" in todo:
    train, valid, test = load_dataset('ule', dtype='uint8', permute_train=True)
    write_dataset('ule', train, valid, test)
    train = scipy.sparse.csr_matrix(train)
    valid = scipy.sparse.csr_matrix(valid)
    test = scipy.sparse.csr_matrix(test)
    write_dataset('ule', train, valid, test, pickle=True)
    print "You must manually compress the created file with gzip!"
