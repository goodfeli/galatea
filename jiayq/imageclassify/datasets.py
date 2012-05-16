'''
dataset.py is a wrapper for loading misc image dataset easier

Yangqing Jia, UC Berkeley EECS
jiayq@eecs.berkeley.edu
'''
import os
import numpy as np
from scipy import misc
import exceptions

def lena():
    '''
    lena: load the raw lena image
    '''
    return misc.imread(os.path.join(os.path.dirname(__file__), 'lena.png'))

class Dataset(object):
    '''
    this is the general interface for datasets
    '''
    # necessary variables:
    # Ntrain: number of training data
    # Ntest: number of testing data
    Ntrain = 0
    Ntest = 0

    # optional (dataset dependent):
    # data_tr: training data
    # data_te: testing data
    # label_tr: training label
    # label_te: testing label
    # imageDim: the image dimension. Should be 3-dimensional for color and 2-dimensional for grayscale
    # Dim: number of raw image feature vector dimension
    data_tr = None
    data_te = None
    label_tr = None
    label_te = None
    imageDim = None
    Dim = None

    def image(self,idx,fromTraining = True):
        '''
        returns an numpy array that can be passed to pyplot.imshow()
        '''
        raise exceptions.NotImplementedError, 'Implement this to return images'

class NdImageDataset(Dataset):
    def __init__(self):
        pass

    @staticmethod
    def fromMatrix(mat, label, imageDim, isTraining=True):
        self = NdImageDataset()
        if isTraining:
            self.data_tr = mat
            self.label_tr = label
            self.Ntrain = mat.shape[0]
        else:
            self.data_te = mat
            self.label_te = label
            self.Ntest = mat.shape[0]
        self.imageDim = imageDim
        self.Dim = mat.shape[1]
        return self

    def image(self,idx,fromTraining = True):
        if fromTraining:
            return self.data_tr[idx].reshape(self.imageDim)
        else:
            return self.data_te[idx].reshape(self.imageDim)

class PatchDataset(Dataset):
    def __init__(self,data_tr,shape):
        self.data_tr = data_tr
        self.Ntrain = data_tr.shape[0]
        self.shape = shape
            
    def image(self,idx,fromTraining=True):
        return self.data_tr[idx].reshape(self.shape)
    
class FolderDataset(Dataset):
    '''
    this is a wrapper that builds a dataset-like structure from a flat folder of images
    '''
    def __init__(self,rootfolder,file_ext,reader=misc.imread,prefetch=True):
        self.rootfolder = os.path.abspath(rootfolder)
        self.reader = reader
        self.prefetch = prefetch
        files = [file for file in os.listdir(rootfolder) if file.lower().endswith(file_ext)]
        self.data = []
        if prefetch:
            for file in files:
                self.data.append(reader(os.path.join(rootfolder,file)))
        self.files = files
        self.Ntrain = len(files)
        self.Ntest = 0

    def image(self,idx,fromTraining=True):
        if self.prefetch:
            return self.data[idx]
        else:
            return self.reader(os.path.join(self.rootfolder,self.files[idx]))
        
class CifarDataset(Dataset):
    '''
    these are constants for CIFAR
    '''
    Ntrain = 50000
    batchsize = 10000
    Ntest = 10000
    Dim = 3072
    imageDim = [32,32,3]
    
    def unpickle(self,filename):
        '''
        the unpickle function from
        http://www.cs.toronto.edu/~kriz/cifar.html
        '''
        import cPickle
        fo = open(filename,'rb')
        data = cPickle.load(fo)
        fo.close()
        return data
    
    def __call__(self, idx, fromTraining = True):
        return self.image(idx, fromTraining)
    
    def image(self,idx,fromTraining = True):
        if fromTraining:
            datum = self.data_tr[idx]
        else:
            datum= self.data_te[idx]
        im = np.empty(self.imageDim, dtype=self.dtype)
        im[:,:,0] = datum[:1024].reshape(32,32)
        im[:,:,1] = datum[1024:2048].reshape(32,32)
        im[:,:,2] = datum[2048:].reshape(32,32)
        return im
    
    def __init__(self,rootfolder = None, dtype='uint8', isTestonly = False):
        # some special cases for Yangqing's personal use
        if rootfolder is None or rootfolder == 'icsi':
            # use the icsi path
            rootfolder = '/u/vis/x1/common/CIFAR/cifar-10-batches-py'
        if rootfolder == 'mac':
            rootfolder = '/Users/jiayq/Research/datasets/cifar-10-batches-py'
        self.dtype = dtype
        if os.path.exists(rootfolder+os.sep+'batches.meta'):
            self.name='CIFAR-10'
            self.load_cifar10(rootfolder,dtype,isTestonly)
        elif os.path.exists(rootfolder+os.sep+'meta'):
            self.name='CIFAR-100'
            self.load_cifar100(rootfolder, dtype, isTestonly)
        else:
            raise exceptions.IOError, 'Cannot understand the dataset format.'
    
    def load_cifar100(self,rootfolder,dtype,isTestonly):
        self.specs = self.unpickle(rootfolder+os.sep+'meta')
        if not isTestonly:
            self.data_tr = np.empty((self.Ntrain,self.Dim),dtype=dtype)
            self.label_tr = np.empty(self.Ntrain,dtype=dtype)
            # just aliases
            self.finelabel_tr = self.label_tr
            self.coarselabel_tr = np.empty(self.Ntrain,dtype=dtype)
            batch = self.unpickle(rootfolder+os.sep+'train')
            self.data_tr[:] = batch['data']
            self.label_tr[:] = np.array(batch['fine_labels'])
            self.coarselabel_tr[:] = np.array(batch['coarse_labels'])
        
        self.data_te = np.empty((self.Ntest,self.Dim),dtype=dtype)
        self.label_te = np.empty(self.Ntest,dtype=dtype)
        self.finelabel_te = self.label_te
        self.coarselabel_te = np.empty(self.Ntest,dtype=dtype)
        batch = self.unpickle(rootfolder+os.sep+'test')
        self.data_te[:] = batch['data']
        self.label_te[:] = np.array(batch['fine_labels'])
        self.coarselabel_te[:] = np.array(batch['coarse_labels'])
    
    def load_cifar10(self,rootfolder,dtype,isTestonly):
        self.specs = self.unpickle(rootfolder+os.sep+'batches.meta')
        if not isTestonly:
            self.data_tr = np.empty((self.Ntrain,self.Dim),dtype=dtype)
            self.label_tr = np.empty(self.Ntrain,dtype=dtype)
            # training batches
            for i in range(5):
                batch = self.unpickle('{}{}data_batch_{}'.format(rootfolder,os.sep,i+1))
                self.data_tr[self.batchsize*i:self.batchsize*(i+1)] = batch['data']
                self.label_tr[self.batchsize*i:self.batchsize*(i+1)] = np.array(batch['labels'])
            
        self.data_te = np.empty((self.Ntest,self.Dim),dtype=dtype)
        self.label_te = np.empty(self.Ntest,dtype=dtype)
        #testing batches
        batch = self.unpickle(rootfolder+os.sep+'test_batch')
        self.data_te[:] = batch['data']
        self.label_te[:] = np.array(batch['labels'])


class CifarDatasetMirror(CifarDataset):
    '''
    This is an extension of the Cifar dataset by mirroring the Cifar dataset horizontally 
    to create 10,000 training data. (Testing data is not mirrored).
    '''
    def __init__(self,rootfolder = None, dtype='uint8', isTestonly = False):
        CifarDataset.__init__(self,rootfolder,dtype,isTestonly)
        self.Ntrain_base = self.Ntrain
        self.Ntrain *= 2
        self.label_tr = np.hstack((self.label_tr,self.label_tr))

    def __call__(self, idx, fromTraining = True):
        return self.image(idx, fromTraining)
    
    def image(self,idx,fromTraining = True):
        if fromTraining:
            datum = self.data_tr[idx % self.Ntrain_base]
        else:
            datum= self.data_te[idx]
        im = np.empty(self.imageDim, dtype=self.dtype)
        im[:,:,0] = datum[:1024].reshape(32,32)
        im[:,:,1] = datum[1024:2048].reshape(32,32)
        im[:,:,2] = datum[2048:].reshape(32,32)
        if fromTraining and np.int(idx / self.Ntrain_base):
            im = im[:,::-1,:]
        return im

class MNISTDataset(Dataset):
    Ntrain = 60000
    Ntest = 10000
    Dim = 784
    imageDim = [28,28]

    def image(self,idx,fromTraining = True):
        if fromTraining:
            return self.data_tr[idx].reshape(self.imageDim)
        else:
            return self.data_te[idx].reshape(self.imageDim)
    
    def __init__(self,rootfolder = None, isTestonly = False):
        if rootfolder is None or rootfolder == 'icsi':
            # icsi folder
            rootfolder = '/u/vis/x1/common/mnist/'
        self.data_tr = self.read_byte_data(os.path.join(rootfolder,'train-images-idx3-ubyte'), 16,[self.Ntrain,self.Dim])
        self.label_tr = self.read_byte_data(os.path.join(rootfolder,'train-labels-idx1-ubyte'), 8, [self.Ntrain])
        self.data_te = self.read_byte_data(os.path.join(rootfolder,'t10k-images-idx3-ubyte'), 16,[self.Ntest,self.Dim])
        self.label_te = self.read_byte_data(os.path.join(rootfolder,'t10k-labels-idx1-ubyte'), 8, [self.Ntest])
    
    def read_byte_data(self,filename, skipbytes, shape):
        fid = open(filename,'rb')
        fid.seek(skipbytes)
        nbytes = np.prod(shape)
        rawdata = fid.read(nbytes)
        fid.close()
        #convert rawdata to data
        data = np.zeros(nbytes)
        for i in range(nbytes):
            data[i] = ord(rawdata[i])
        data.resize(shape)
        return data

class MNISTDatasetSub(MNISTDataset):
    '''
    MNISTDatasetSub hosts the MNIST dataset, only containing the digits
    specified by the digits parameter during initialization.
    '''
    def __init__(self,digits,rootfolder=None,isTestOnly=False):
        '''
        __init__: loades the data.
            digits: the digits you want to include in the dataset, e.g. [4,9]
        '''
        MNISTDataset.__init__(self,rootfolder,isTestOnly)
        # just in case you pass in char instead of int
        digits = [int(i) for i in digits]
        self.rawLabel = digits
        # clean those data that are not used
        # training data
        newlabel = -np.ones(self.label_tr.shape,dtype=np.int)
        for i in range(len(digits)):
            newlabel[self.label_tr == digits[i]] = i
        self.data_tr = self.data_tr[newlabel >= 0]
        self.label_tr = newlabel[newlabel >= 0]
        self.Ntrain = self.label_tr.size
        
        # testing data
        newlabel = -np.ones(self.label_te.shape,dtype=np.int)
        for i in range(len(digits)):
            newlabel[self.label_te == digits[i]] = i
        self.data_te = self.data_te[newlabel >= 0]
        self.label_te = newlabel[newlabel >= 0]
        self.Ntest = self.label_te.size
