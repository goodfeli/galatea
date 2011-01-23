""" 
This is a script to transform the data from Unsupervised and Transfer Learning Challenge(UTLC)

The web page is at: http://www.causality.inf.ethz.ch/home.php

dataset:
wget http://www.causality.inf.ethz.ch/ul_data/avicenna_text.zip
wget http://www.causality.inf.ethz.ch/ul_data/harry_text.zip
wget http://www.causality.inf.ethz.ch/ul_data/rita_text.zip
wget http://www.causality.inf.ethz.ch/ul_data/sylvester_text.zip
wget http://www.causality.inf.ethz.ch/ul_data/terry_text.zip
wget http://www.causality.inf.ethz.ch/ul_data/ule_text.zip

TODO: pylearn.io.sparse? That will use numpy.tofile?
TODO: pylearn.datasets/utlc.py to load all of them.

dataset harry and terry are stored in a dump of scipy.special.csr_matrix
other dataset are in pylearn.io.filetensor format.
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
def load_dataset(name, dtype=None, permute_train=False, normalize=False):
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
        train = train[perm,:]
    valid = numpy.loadtxt(os.path.join(ROOT_PATH,name+'_text',
                                       name+'_valid.data'),
                          dtype=dtype)
    test = numpy.loadtxt(os.path.join(ROOT_PATH,name+'_text',
                                      name+'_final.data'),
                         dtype=dtype)
    if normalize:
        mean = train.mean(axis=0)
        std = train.std(axis=0)
        train = train-mean
        valid = valid-mean
        test = test-mean
        for i in range(std.shape[0]):
            if std[i]!=0:
                #histogramme.
                print train[:,i].min(), train[:,i].max(), train[:,i].mean(), train[:,i].std()
                train[:,i]/=std[i]
                #print train[:,i].min(), train[:,i].max(), train[:,i].mean(), train[:,i].std()
                #import pdb;pdb.set_trace()
                valid[:,i]/=std[i]
                test[:,i]/=std[i]
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
        train = train[perm,:]
    return train, valid, test

def load_coo_matrix(name, dtype=None, permute_train=False):
    train, valid, test = load_dataset(name, dtype, permute_train=False)
    
    if permute_train:
        import pdb;pdb.set_trace()
        rng = numpy.random.RandomState([1,2,3])
        perm = rng.permutation(train.shape[0])
        train = train[perm,:,:]
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


#[i.max() for i in [train,valid,test]]
#[i.min() for i in [train,valid,test]]
#[numpy.allclose(i,numpy.asarray(i,dtype='int16')) for i in [train,test,valid]]


if "--avicenna" in todo:
#avicenna: sparsity=0%
#nb feature: 120
#nb row(train,valid,test)=(150205,4096,4096)
#Nb element:(150205+4096+4096)*120/1000./1000=19007640 ~=19M -> int16 ~38Mb, float32 ~76Mb
#range: 0-999 type: int16
    train, valid, test = load_dataset('avicenna', dtype='int16', permute_train=True)
    write_dataset('avicenna', train, valid, test)
    print "You must manually compress the created file with gzip!"


if "--harry" in todo:
#harry_train.ft 665M
#harry_train.ft.gz 17M
#dump of scipy.sparse.csr_matrix: 154M gzip: 20M
#harry: sparsity=98.12%
#nb feature: 5000
#nb row(train,valid,test)=(69652,4096,4096)
#Nb element:(69652+4096+4096)*5000/1000./1000~=389M -> int16 ~780Mb, float32 ~1560Mb
#range: train:0-451 valid:0-19, test:0-24 type: int16
    train, valid, test = load_dataset2('harry', dtype='int16', rows_size=5000, permute_train=True)
    write_dataset('harry', train, valid, test)
    train = scipy.sparse.csr_matrix(train)
    valid = scipy.sparse.csr_matrix(valid)
    test = scipy.sparse.csr_matrix(test)
    write_dataset('harry', train, valid, test, pickle=True)
    print "You must manually compress the created file with gzip!"

if "--rita" in todo:
#DONE, file not gzipped as it is not worth the size saved!
#rita: sparsity=1.19%
#nb feature: 7200
#nb row(train,valid,test)=(111808,4096,4096)
#Nb element:(111808+4096+4096)*7200/1000./1000~=864M -> int16 ~1728Mb, float32 ~3456Mb
#range: train/valid/test 0-230 type: uint8
    train, valid, test = load_dataset2('rita', dtype='uint8', rows_size=7200, permute_train=True)
    write_dataset('rita', train, valid, test)
    print "You MUST NOT manually compress the created file with gzip! They are not smaller"

if "--sylvester" in todo:
#DONE, file not gzipped as it is not worth the size saved!~25% saved
#sylvester: sparsity 0%
#nb feature: 100
#nb row(train,valid,test)=(572820,4096,4096)
#Nb element:(572820+4096+4096)*100/1000./1000~=58M -> int16 ~116Mb, float32 ~232Mb
#range: 0-999 type: int16
    train, valid, test = load_dataset2('sylvester', dtype='int16', rows_size=100, permute_train=True)
    write_dataset('sylvester', train, valid, test)
    print "You must manually compress the created file with gzip!"

if "--terry" in todo:
#terry: sparsity=99.84%
#nb feature: 47236
#nb row(train,valid,test)=(217034,4096,4096)
#Nb element:(217034+4096+4096)*47236/1000./1000~=10368M -> int16 ~21Gb, float32 ~43Gb
#range: train:2-999, test:2-684, valid:3-728 type: int16
#terry_train.npy 605M
    train, valid, test = load_coo_matrix('terry', dtype='float32', permute_train=True)#, rows_size=47236)
    write_dataset('terry', train, valid, test, pickle=True)
    print "You must manually compress the created file with gzip!"

if "--ule" in todo:
#ule: sparsity=80.85%
#nb feature: 784
#nb row(train,valid,test)=(26808,4096,4096)
#Nb element:(26808+4096+4096)*784/1000./1000~=27M -> uint8 ~27M, int16 ~55Mb, float32 ~110Mb
#range: 0-255 type: uint8
    train, valid, test = load_dataset('ule', dtype='uint8', permute_train=True, normalize=True)
    write_dataset('ule', train, valid, test)
    train = scipy.sparse.csr_matrix(train)
    valid = scipy.sparse.csr_matrix(valid)
    test = scipy.sparse.csr_matrix(test)
    write_dataset('ule', train, valid, test, pickle=True)
    print "You must manually compress the created file with gzip!"
