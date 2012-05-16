import numpy as np
from matplotlib import cm,pyplot
from jiayq.utils import gemm, fastop, mpiutils
from jiayq import kmeans
from scipy import io
import exceptions
import datasets

class PatchExtractor:
    def __init__(self, psize, nChannels = 1, normalize=None, specs = {}, stride = 1):
        self.nChannels = nChannels
        if type(psize) is int:
            psize = [psize,psize]
        self.psize = psize
        self.normalize = normalize
        self.specs = specs
        self.Dim = np.prod(self.psize) * nChannels
        self.stride = stride

    def sample(self, dataset, nPatches, returnPosition=False, imids=None,rowids=None,colids=None):
        patches = np.empty((nPatches, self.Dim))
        if imids is None or rowids is None or colids is None:
            self.imids = np.random.randint(dataset.Ntrain,size=nPatches)
            if dataset.imageDim is not None:
                if dataset.imageDim[0]-self.psize[0] == 0:
                    self.rowids = np.zeros(nPatches,dtype=np.int)
                else:
                    self.rowids = np.random.randint(dataset.imageDim[0]-self.psize[0], size=nPatches)
                if dataset.imageDim[1]-self.psize[1] == 0:
                    self.colids = np.zeros(nPatches,dtype=np.int)
                else:
                    self.colids = np.random.randint(dataset.imageDim[1]-self.psize[1], size=nPatches)
                precomputed = True
            else:
                self.rowids = np.zeros(nPatches,dtype=np.int)
                self.colids = np.zeros(nPatches,dtype=np.int)
                precomputed = False
        else:
            self.imids = imids
            self.rowids = rowids
            self.colids = colids
            precomputed = True

        for i in range(nPatches):
            im = dataset.image(self.imids[i])
            if not precomputed:
                self.rowids[i] = np.random.randint(im.shape[0]-self.psize[0])
                self.colids[i] = np.random.randint(im.shape[1]-self.psize[1])
            patch = im[self.rowids[i]:self.rowids[i]+self.psize[0], self.colids[i]:self.colids[i]+self.psize[1]]
            patches[i] = patch.flat
        self.normalizePatches(patches)
        if returnPosition:
            return patches,np.vstack((self.rowids,self.colids))
        else:
            return patches

    def denseExtract(self, image, positionNormalize = True):
        '''
        dense extraction of image patches with a given stride
        returns the patches and the relative positions, and
        the (height,width) pair that can be used
        to reconstruct the patches to a height*width*Dim cube
        '''
        imheight = image.shape[0]
        imwidth = image.shape[1]
        stride = self.stride
        idxh = range(0,imheight-self.psize[0]+1,stride)
        idxw = range(0,imwidth-self.psize[1]+1,stride)
        nPatches= len(idxh)*len(idxw)
        if nPatches == 0:
            return np.array([])
        patches = np.empty((nPatches,self.Dim))
        curr = 0
        for i in idxh:
            for j in idxw:
                patches[curr] = image[i:i+self.psize[0],j:j+self.psize[1]].flat
                curr += 1
        self.normalizePatches(patches)
        if positionNormalize:
            positions = np.vstack( ((np.tile(idxh,(len(idxw),1)).T.flatten()+0.5) / np.float(imheight-self.psize[0]+1),\
                                (np.tile(idxw,len(idxh))+0.5) / np.float(imwidth-self.psize[0]+1) ) )
        else:
            positions = np.vstack( (np.tile(idxh,(len(idxw),1)).T.flatten(),\
                                np.tile(idxw,len(idxh))) )
        return patches, positions, (len(idxh), len(idxw))

    def normalizePatches(self,patches):
        if self.normalize == 'meanvar':
            # subtract the mean, and normalize the contrast
            if 'reg' in self.specs.keys():
                reg = self.specs['reg']
            else:
                # default parameter
                reg = 10.0
            patches -= np.mean(patches,axis=1).reshape((patches.shape[0],1))
            patches /= (np.sqrt(np.mean(patches**2,axis=1))+reg).reshape((patches.shape[0],1))
        elif self.normalize == 'unitball':
            try:
                reg = self.specs['reg']
            except KeyErrorw:
                # default parameter
                # we set this to the default parameter as used by
                # Andrew Ng in his AISTATS10 paper.
                reg = 10.0
            # move everything onto the unit ball
            patches /= (np.sqrt(np.mean(patches**2,axis=1))+reg).reshape((patches.shape[0],1))
        else:
            # do nothing
            pass

class PatchVisualizer:
    '''
    PatchVisualizer visualizes patches or codes.
    Use like this:
       vis = PatchVisualizer()
       vis(patches)
    patches can be a vector (one patch), or a 2-dim matrix
    each row being a patch. The dimension (and color) will
    be automatically identified.
    '''
    gap = 1
    def __call__(self, patches, newfig = True, stop=False, title=None, bgColorFunc = np.mean, outputFile = None):
        if not stop:
            pyplot.ion()
        if newfig:
            pyplot.figure()
        if len(patches.shape) == 1:
            # we are given only a single patch
            image = self.getSingleImage(patches)
        else:
            # we are given a set of patches
            nPatches = patches.shape[0]
            nPatchesPerEdge = int(np.ceil(np.sqrt(nPatches)))
            pSize = self.getPatchShape(patches[0])
            pSizeExpand = list(pSize)
            pSizeExpand[0] += self.gap
            pSizeExpand[1] += self.gap
            imSize = list(pSizeExpand)
            imSize[0] = imSize[0] * nPatchesPerEdge - self.gap
            imSize[1] = imSize[1] * nPatchesPerEdge - self.gap
            image = np.ones(imSize) * bgColorFunc(patches)
            for pid in range(nPatches):
                rowid = pid / nPatchesPerEdge
                colid = pid % nPatchesPerEdge
                image[rowid*pSizeExpand[0]:rowid*pSizeExpand[0]+pSize[0], \
                      colid*pSizeExpand[1]:colid*pSizeExpand[1]+pSize[1]] = self.getSingleImage(patches[pid])
        image -= np.min(image)
        image /= np.max(image)
        if image.ndim==2:
            # I have no idea why eclipse reports an error, but never mind.
            pyplot.imshow(image, cmap=cm.gray, interpolation='nearest',)
        else:
            pyplot.imshow(image, interpolation='nearest')
        if title:
            pyplot.title(title)
        #pyplot.xticks([])
        #pyplot.yticks([])
        pyplot.axis('off')
        pyplot.draw()
        if outputFile is not None:
            pyplot.savefig(outputFile)
        if stop:
            print 'Waiting for figure...'
            pyplot.ioff()
            pyplot.show()

    def getSingleImage(self,patch, toGrey=False):
        patch = patch.reshape(self.getPatchShape(patch))
        if toGrey and patch.ndim == 3:
            patch = np.mean(patch,axis=2)
        return patch

    def togrey(self, patches):
        shape = self.getPatchShape(patches[0])
        if len(shape) == 2:
            return patches
        else:
            patches.resize(np.hstack((patches.shape[0],shape)))
            patches = np.mean(patches,axis=3)
            patches.resize((patches.shape[0],np.prod(patches.shape[1:])))
            return patches

    def fromMatlabFormat(self,patches):
        shape = self.getPatchShape(patches[0])
        if len(shape) == 2:
            return patches
        else:
            size = shape[0]*shape[1]
            patches = patches.reshape((patches.shape[0],shape[2],size))
            patches = patches.swapaxes(1,2)
            return patches.reshape((patches.shape[0],np.prod(shape)))

    def getPatchShape(self,patch):
        edgeLen = np.sqrt(patch.size)
        if edgeLen != np.floor(edgeLen):
            # we are given color patches
            edgeLen = int(np.sqrt(patch.size / 3))
            return [edgeLen,edgeLen,3]
        else:
            edgeLen = int(edgeLen)
            return [edgeLen,edgeLen]

    def draw(self):
        pyplot.draw()

    def barrier(self):
        pyplot.ioff()
        pyplot.show()

patchVisualizer = PatchVisualizer()

class PatchPreprocessor:
    def __init__(self, method, specs = {}, previousPrep = None):
        # previousPrep enables us to have a chain of preprocessors
        self.method = method
        self.specs = specs
        self.previousPrep = previousPrep

    def train_pca_zca(self,patches):
        self.b = -np.mean(patches,axis=0)
        patches -= self.b
        try:
            whitenReg = self.specs['whitenReg']
        except KeyError:
            # this is the regularization term used in
            # Andrew Ng's AISTATS paper
            whitenReg = 0.1
        covmat = np.cov(patches, rowvar=0)
        [eigval,eigvec] = np.linalg.eigh(covmat)
        self.W = eigvec * 1.0 / (np.sqrt(np.maximum(eigval, 0.0))+ whitenReg)
        if self.method == 'zca':
            self.W = np.dot(self.W, eigvec.T)

    def train_gaussdiscount(self,patches):
        try:
            shape = self.specs['patchshape']
        except KeyError:
            raise KeyError, "Error: you should provide the shape of the patch"
        # precompute the discount map
        size = shape[0]
        center = size/2.0-0.5
        if 'sigma' in self.specs.keys():
            sigma2 = self.specs['sigma']**2
        else:
            sigma2 = (size / 2.0 / np.sqrt(2.0*np.log(2.0)))**2
        W = np.exp(-(np.array(range(size))-center) ** 2 / 2 / sigma2)
        # make 2d
        W = (W.reshape(size,1) * W).flatten()
        self.W = W / np.max(W)
        if len(shape) == 3:
            # we have a color patch
            self.W = np.tile(self.W, (shape[2],1)).T.flatten()

    def train(self,patches):
        if self.previousPrep is not None:
            self.previousPrep.train(patches)
            patches = self.previousPrep.process(patches)
        # we will deal with patches using double precision.
        patches = patches.astype(np.float64)
        self.Dim = patches.shape[1]
        if self.method == 'identical' or self.method == 'fft' or \
           self.method == 'fftmag2' or self.method =='fftflat':
            # these methods has nothing to train
            pass
        elif self.method == 'pca' or self.method == 'zca':
            self.train_pca_zca(patches)
        elif self.method == 'gaussdiscount':
            self.train_gaussdiscount(patches)
        else:
            raise exceptions.NotImplementedError, "PatchPreprocessor: method {} Not Implemented.".format(self.method)

    def process(self,patches):
        if self.previousPrep is not None:
            self.previousPrep.process(patches)
        # process
        if self.method == 'identical':
            return patches.copy()
        elif self.method == 'fft' or self.method == 'fftmag2' or self.method == 'fftflat':
            size = patchVisualizer.getPatchShape(patches[0])[0]
            output = np.empty((patches.shape[0],size*size), dtype=np.complex)
            for i in range(patches.shape[0]):
                output[i] = np.fft.fftshift(np.fft.fft2(patchVisualizer.getSingleImage(patches[i], toGrey=True))).flatten()
            if self.method == 'fftmag2':
                output = np.abs(output)**2
            if self.method == 'fftflat':
                output = np.hstack((np.real(output), np.imag(output)))
            return output
        elif self.method == 'pca' or self.method == 'zca':
            return np.dot(patches,self.W) + self.b
        elif self.method == 'gaussdiscount':
            return patches*self.W
        else:
            raise exceptions.NotImplementedError, "PatchPreprocessor: method {} Not Implemented.".format(self.method)

class DictTrainer:
    def __init__(self, K, method, specs={}):
        self.K = K
        self.method = method
        self.specs = specs

    def train(self,patches):
        patches = patches.astype(np.float64)
        if self.method == 'kmeans':
            # k-means
            maxiter = 100
            verbose = 1
            ninit = 1
            if 'maxiter' in self.specs.keys():
                maxiter = self.specs['maxiter']
            if 'verbose' in self.specs.keys():
                verbose = self.specs['verbose']
            if 'ninit' in self.specs.keys():
                ninit = self.specs['ninit']
            self.dictionary = kmeans.kmeans(patches, self.K, n_init = ninit, max_iter=maxiter, verbose=verbose)[0]
        elif self.method == 'random':
            idx = np.array(range(patches.shape[0]))
            np.random.shuffle(idx)
            self.dictionary = patches[idx[:self.K]].copy()
        else:
            raise exceptions.NotImplementedError, "Not Implemented."
        return self.dictionary

class PatchEncoder:
    def __init__(self, method, specs={}, dictionary = None):
        self.dictionary = dictionary
        self.method = method
        self.specs = specs

    def encode(self,patches):
        if self.dictionary is None:
            raise exceptions.ValueError, "You should first provide a dictionary."
        if self.method == 'inner':
            encoded = gemm.mydot(patches, self.dictionary.T)
        elif self.method == 'thres' or self.method == 'thres_single':
            innerproduct = gemm.mydot(patches, self.dictionary.T)
            try:
                alpha = self.specs['alpha']
            except KeyError:
                # Andrew Ng's standard parameter
                alpha = 0.25
            if self.method == 'thres':
                encoded = np.hstack((np.maximum(0, innerproduct-alpha),\
                                     np.maximum(0, -innerproduct-alpha)))
            else:
                encoded = np.maximum(0,innerproduct-alpha)
        elif self.method == 'tri':
            distance = kmeans.euclidean_distances(patches,self.dictionary)
            mu = np.mean(distance,axis=1)
            encoded = np.maximum(0.0,mu.reshape(mu.size,1)-distance)
        else:
            raise exceptions.NotImplementedError, "Not Implemented."
        return encoded

class SpatialPooler:
    '''
    the class that does spatial pooling
    '''
    def __init__(self, grid, method, spec={}):
        self.grid = grid
        if type(grid) is int:
            self.grid = [grid,grid]
        self.nBins = np.prod(self.grid)
        self.method = method
        self.spec = spec

    def pool(self, activations, positions):
        hwbins = (positions * np.array(self.grid).reshape(2,1)).astype(np.int)
        bins = hwbins[0]*self.grid[1] + hwbins[1]
        if self.method == 'max':
            output = fastop.fastmaximums(activations, bins, self.nBins)[0]
        elif self.method == 'ave':
            output = fastop.fastcenters(activations, bins, self.nBins)[0]
        else:
            raise exceptions.NotImplementedError, 'Not Implemented.'
        return output

class Pipeliner:
    # the parameters are set to be Andrew Ng's default parameters
    def __init__(self, extractor = PatchExtractor(6,nChannels=3,normalize='meanvar'), \
                 preprocessor = PatchPreprocessor('zca'),\
                 dictTrainer = DictTrainer(1600, method='omp', specs={'maxiter':50}),\
                 encoder = PatchEncoder('thres'),\
                 pooler = SpatialPooler(2,'ave')):
        self.extractor = extractor
        self.preprocessor = preprocessor
        self.dictTrainer = dictTrainer
        self.encoder = encoder
        self.pooler = pooler

    def train(self,dataset,nPatches):
        if self.preprocessor is not None:
            if mpiutils.rank == 0:
                # we train preprocessor (e.g. zca) on the root node only
                patches = self.extractor.sample(dataset,nPatches)
                self.preprocessor.train(patches)
            mpiutils.safebarrier()
            self.preprocessor = mpiutils.comm.bcast(self.preprocessor,root=0)
            if self.encoder is not None and self.dictTrainer is not None:
                patches = self.extractor.sample(dataset,int(nPatches / mpiutils.size))
                patches = self.preprocessor.process(patches)
                self.encoder.dictionary = self.dictTrainer.train(patches)
        mpiutils.safebarrier()

    def process_single(self, image):
        patches, positions = self.extractor.denseExtract(image)[:2]
        patches = self.preprocessor.process(patches)
        activations = self.encoder.encode(patches)
        pooled = self.pooler.pool(activations, positions)
        return pooled

    def process_dataset(self, dataset, fromTraining = True, start = 0, end = None):
        '''
        process the whole dataset. Warning: may be very time-consuming.
        '''
        if end is None:
            if fromTraining:
                end = dataset.Ntrain
            else:
                end = dataset.Ntest
        # run the first image to obtain the pooled feature size
        pooled = self.process_single(dataset.image(0, fromTraining))
        # due to legacy reasons, data is a Nimage*Nfeatures matrix
        # the features are sorted as
        # [bin1feat1,bin1feat2....bin2feat1,bin2feat2....binMfeatN]
        data = np.empty((end-start,pooled.size))
        for idx in range(start,end):
            data[idx-start] = self.process_single(dataset.image(idx, fromTraining)).flatten()
        return data

    def batch_process_dataset(self, dataset, batchsize = 1000, filename_template = '{}_{}.mat',
                              fromTraining = True):
        '''
        process the whole dataset. Warning: may be very time-consuming.
        '''
        if fromTraining:
            Nimages = dataset.Ntrain
        else:
            Nimages = dataset.Ntest

        Nbatches = (Nimages+batchsize-1) / batchsize
        for batchid in range(mpiutils.rank, Nbatches, mpiutils.size):
            mpiutils.nodeprint('Batch {} of {}'.format(batchid, Nbatches))
            start = batchsize * batchid
            end = min(start+batchsize, Nimages)
            feat = Pipeliner.process_dataset(self, dataset, fromTraining, start, end)
            io.savemat(filename_template.format(batchsize, batchid),\
                       {'feat':feat}, oned_as='rows')

