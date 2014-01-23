#! /usr/bin/env ipythonpl

import pdb
from numpy import array, dot, random, linalg, sqrt, asarray, cov, eye, sum, hstack
from numpy.linalg import norm

from cache import cached, PersistentHasher



class PCA(object):
    def __init__(self, xx):
        '''
        Inspired by PCA in matplotlib.mlab

        Compute the principle components of a dataset and stores the
        mean, sigma, and SVD of sigma for the data.  Use toPC and
        fromPC to project the data onto a reduced set of dimensions
        and back. This version takes the SVD of the covariance matrix.

        Inputs:

          *xx*: a numobservations x numdims array

        Attrs:

          *nn*, *mm*: the dimensions of xx

          *mu* : a numdims array of means of xx

          *sigma* : the covariance matrix

          *var* : the average amount of variance of each of the principal components

          *std* : sqrt of var

          *fracVar* : the fractional amount of variance from each principal component

          *fracStd* : sqrt of fracVar
        '''

        self.nn, self.mm = xx.shape
        if self.nn < self.mm:
            raise RuntimeError('we assume data in a is organized with numrows>numcols')

        self.mu          = xx.mean(axis=0)
        centeredXX       = self.center(xx)
        #self.sigma       = dot(centeredXX.T, centeredXX) / self.nn
        self.sigma       = cached(dot, centeredXX.T, centeredXX) / self.nn

        # Columns of UU are the eigenvectors of self.sigma, i.e. the
        # principle components. UU and VV are transpose of each other;
        # we don't use VV. ss is the diagonal of the true S matrix.
        #self.UU, self.ss, self.VV = linalg.svd(self.sigma, full_matrices = False)
        self.UU, self.ss, self.VV = cached(linalg.svd, self.sigma, full_matrices = False)

        self.var = self.ss / float(self.nn)
        self.std = sqrt(self.var)
        self.fracVar = self.var / self.var.sum()
        self.fracStd = self.std / self.std.sum()


    def pc(self, numDims = None):
        '''Return a matrix whose columnts are the ordered principle
        components.'''

        if numDims is None:
            numDims = self.UU.shape[1]

        return self.UU[:,0:numDims]


    def meanAndPc(self, numDims = None):
        '''Returns a matrix whose first column is the mean and
        subsequent columns are the ordered principle components.'''
        return hstack((array([self.mu]).T, self.pc(numDims = numDims)))


    def toPC(self, xx, numDims = None, whiten = False, epsilon = 0, center = True):
        '''Center the xx and project it onto the principle components.

        Called \tilde{x} on UFLDL wiki page.'''

        xx = asarray(xx)

        if xx.shape[-1] != self.mm:
            raise ValueError('Expected an array with dims[-1] == %d' % self.mm)

        if not whiten and epsilon != 0:
            raise Exception('Probable misuse: epsilon != 0 but whitening is off.')

        if numDims is None:
            numDims = xx.shape[-1]

        if center:
            xx = self.center(xx)

        pc = dot(xx, self.UU[:,0:numDims])

        if whiten:
            pc /= sqrt(self.ss[0:numDims] + epsilon)

        return pc


    def toWhitePC(self, xx, numDims = None, epsilon = 0):
        return self.toPC(xx, numDims = numDims, whiten = True, epsilon = epsilon)
    

    def fromPC(self, pc, unwhiten = False, epsilon = 0, uncenter = True):
        '''Project the given principle components back to the original
        space and uncenter them.'''

        pc = asarray(pc)

        numDims = pc.shape[1]

        if not unwhiten and epsilon != 0:
            raise Exception('Probable misuse: epsilon != 0 but unwhitening is off.')

        if unwhiten:
            xx = dot(pc * sqrt(self.ss[0:numDims] + epsilon), self.UU[:,0:numDims].T)
        else:
            xx = dot(pc, self.UU[:,0:numDims].T)

        if uncenter:
            xx = self.uncenter(xx)

        return xx


    def fromWhitePC(self, pc, epsilon = 0):
        '''Reconstructs data from white PC.'''
        
        return self.fromPC(pc, unwhiten = True, epsilon = epsilon)


    def pcaAndBack(self, xx, numDims = None, error = False):
        '''Projects to first numDims of pca dimensions and back'''

        #print 'U   * U   - I:', norm(dot(self.UU, self.UU)-eye(self.UU.shape[0]))
        #print 'U   * U.T - I:', norm(dot(self.UU, self.UU.T)-eye(self.UU.shape[0]))
        #print 'U.T * U.T - I:', norm(dot(self.UU.T, self.UU.T)-eye(self.UU.shape[0]))
        #print 'U.T * U   - I:', norm(dot(self.UU.T, self.UU)-eye(self.UU.shape[0]))
        #print 'V   * V   - I:', norm(dot(self.VV, self.VV)-eye(self.VV.shape[0]))
        #print 'V   * V.T - I:', norm(dot(self.VV, self.VV.T)-eye(self.VV.shape[0]))
        #print 'V.T * V.T - I:', norm(dot(self.VV.T, self.VV.T)-eye(self.VV.shape[0]))
        #print 'V.T * V   - I:', norm(dot(self.VV.T, self.VV)-eye(self.VV.shape[0]))
        #print 'U   * V   - I:', norm(dot(self.UU, self.VV)-eye(self.UU.shape[0]))
        #print 'U   * V.T - I:', norm(dot(self.UU, self.VV.T)-eye(self.UU.shape[0]))
        #print 'U.T * V.T - I:', norm(dot(self.UU.T, self.VV.T)-eye(self.UU.shape[0]))
        #print 'U.T * V   - I:', norm(dot(self.UU.T, self.VV)-eye(self.UU.shape[0]))
        #pdb.set_trace()
        
        pc = self.toPC(xx, numDims = numDims)
        ret = self.fromPC(pc)
        if error:
            return ret, norm(xx - ret)
        else:
            return ret


    def toZca(self, xx, numDims = None, epsilon = 0):
        '''Return Zero-phase whitening filter version.'''

        pc = self.toWhitePC(xx, numDims = numDims, epsilon = epsilon)
        return self.fromPC(pc, uncenter = False)


    def fromZca(self, zc, numDims = None, epsilon = 0):
        '''Return Zero-phase whitening filter version.'''

        # Computes dot(dot(zca1, pca.UU) * sqrt(pca.ss + 0), pca.UU.T) + pca.mu
        pc = self.toPC(zc, center = False) # already centered
        return self.fromWhitePC(pc, epsilon = epsilon)


    def zcaAndBack(self, xx, numDims = None, epsilon = 0, error = False):
        '''Projects to first numDims of zca dimensions and back. Same
        as pcaAndBack, save for some numerica instability introduced
        by using epsilon > 0. Provided for consistency, but not really
        recommended for use.'''

        print 'You should probably use pcaAndBack instead.'

        zc = self.toZca(xx, numDims = numDims, epsilon = epsilon)
        ret = self.fromZca(zc, numDims = numDims, epsilon = epsilon)
        if error:
            return ret, norm(xx - ret)
        else:
            return ret


    def center(self, xx):
        '''Center the data using the mean from the training set. Does
        not ensure that each dimension has std = 1.'''

        return xx - self.mu


    def uncenter(self, cc):
        '''Undo the operation of center'''

        return cc + self.mu


    def __hash__(self):
        hasher = PersistentHasher()
        hasher.update('PCA')
        hasher.update(self.nn)
        hasher.update(self.mm)
        hasher.update(self.mu)
        hasher.update(self.sigma)
        hasher.update(self.UU)
        hasher.update(self.ss)
        hasher.update(self.VV)
        hasher.update(self.var)
        hasher.update(self.std)
        hasher.update(self.fracVar)
        hasher.update(self.fracStd)
        return int(hasher.hexdigest(), 16)


    def __cmp__(self, other):
        return self.__hash__() - other.__hash__()


def testPca(PcaClass = 'specify a class to use'):
    if PcaClass.__name__ == 'PCA':
        usePcaSvd = False
    elif PcaClass.__name__ == 'PCA_SVD':
        usePcaSvd = True
    else:
        raise Exception('Unknown class')
    
    from matplotlib import pyplot
    random.seed(1)
    
    NN = 10
    #transform = array([[2, 3.5], [3.5, 8]])
    transform = array([[5, 3.5, .3], [3.5, 8, 7], [4, 2, 9]])
    #transform = array([[2, 4.5], [2.5, 8]])
    data1 = random.randn(NN,3)
    data1 = dot(data1, transform)
    data1[:,0] += 4
    data1[:,1] += -2
    data1[:,2] += 1
    data2 = random.randn(NN,3)
    data2 = dot(data2, transform)
    data2[:,0] += 4
    data2[:,1] += -2
    data2[:,2] += 1

    print 'data1\n', data1
    print 'data2\n', data2
    print

    if usePcaSvd:
        print '\nPCA_SVD (OLD VERSION)'
        testPcaHelper(data1, data2, PcaClass, usePcaSvd = True)
    else:
        print '\nPCA'
        testPcaHelper(data1, data2, PcaClass, usePcaSvd = False)

    pyplot.show()



def testPcaHelper(data1, data2, PcaClass, usePcaSvd):
    '''Helper function for testing either PCA_SVD or PCA.'''

    from matplotlib import pyplot

    pca = PcaClass(data1)

    print 'Principle components (columns)\n', pca.pc()
    print 'data1 centered\n',       pca.center(data1)
    print 'data1 uncentered\n',     pca.uncenter(pca.center(data1))
    print 'data1 toPC\n',           pca.toPC(data1)
    print 'data1 cov(toPC.T)\n',    cov(pca.toPC(data1).T, bias = 1)
    if not usePcaSvd:
        print 'data1 cov(toWhitePC.T)\n',    cov(pca.toWhitePC(data1).T, bias = 1)
    print 'data1 fromPC\n',         pca.fromPC(pca.toPC(data1))
    print 'data1 toPC (1 dim)\n',   pca.toPC(data1, 1)
    print 'data1 fromPC (1 dim)\n', pca.fromPC(pca.toPC(data1, 1))
    if not usePcaSvd:
        print 'data1 zca\n',    pca.toZca(data1, 1)

    pc1 = pca.toPC(data1)
    if not usePcaSvd:
        pc1white = pca.toWhitePC(data1)
    recon1 = pca.fromPC(pc1)

    pc2 = pca.toPC(data2)
    if not usePcaSvd:
        pc1white = pca.toWhitePC(data1)
        pc1whiteRecon = pca.fromWhitePC(pc1white)
        zca1Manual = pca.fromPC(pc1white, uncenter = False)
        zca1 = pca.toZca(data1)
        zca1Red = pca.toZca(data1, numDims = 5, epsilon = .1)

        fromZca1      = pca.fromZca(zca1)
        fromZca1Red   = pca.fromZca(zca1Red, numDims = 5, epsilon = .1)
        fromZca1Redtf = pca.zcaAndBack(data1, numDims = 5, epsilon = .1)

        pc2white = pca.toWhitePC(data2)
        pc2whiteRecon = pca.fromWhitePC(pc2white)
        zca2Manual = pca.fromPC(pc2white, uncenter = False)
        zca2 = pca.toZca(data2)


    recon2 = pca.fromPC(pc2)

    recon2_1dim = pca.fromPC(pca.toPC(data2, 1))
    recon2_2dim = pca.fromPC(pca.toPC(data2, 2))
    recon2_3dim = pca.fromPC(pca.toPC(data2, 3))

    print 'data 2 recon error, 1 dim', ((recon2_1dim - data2)**2).sum()
    print 'data 2 recon error, 1 dim', ((pca.pcaAndBack(data2, 1) - data2)**2).sum()
    print 'data 2 recon error, 1 dim', ((pca.zcaAndBack(data2, 1) - data2)**2).sum()
    print 'data 2 recon error, 1 dim', ((pca.zcaAndBack(data2, 1, epsilon = 1) - data2)**2).sum()
    print 'data 2 recon error, 2 dim', ((recon2_2dim - data2)**2).sum()
    print 'data 2 recon error, 2 dim', ((pca.pcaAndBack(data2, 2) - data2)**2).sum()
    print 'data 2 recon error, 2 dim', ((pca.zcaAndBack(data2, 2) - data2)**2).sum()
    print 'data 2 recon error, 2 dim', ((pca.zcaAndBack(data2, 2, epsilon = 1) - data2)**2).sum()
    print 'data 2 recon error, 3 dim', ((recon2_3dim - data2)**2).sum()
    print 'data 2 recon error, 3 dim', ((pca.pcaAndBack(data2, 3) - data2)**2).sum()
    print 'data 2 recon error, 3 dim', ((pca.zcaAndBack(data2, 3) - data2)**2).sum()
    print 'data 2 recon error, 3 dim', ((pca.zcaAndBack(data2, 3, epsilon = 1) - data2)**2).sum()

    pyplot.figure()

    pyplot.subplot(3,4,1)
    pyplot.plot(data1[:,0], data1[:,1], 'o')
    str = 'PCA_SVD' if usePcaSvd else 'PCA'
    pyplot.title(str + ': data1')

    pyplot.subplot(3,4,2)
    pyplot.plot(pc1[:,0], pc1[:,1], 'o')
    pyplot.title('pc1')

    pyplot.subplot(3,4,3)
    dat,err = pca.pcaAndBack(data1, error = True)
    print 'Error:', err
    pyplot.plot(dat[:,0], dat[:,1], 'o')
    pyplot.title('pcaAndBack')

    pyplot.subplot(3,4,4)
    pyplot.plot(recon1[:,0], recon1[:,1], 'o')
    pyplot.title('recon1')

    pyplot.subplot(3,4,5)
    pyplot.plot(data2[:,0], data2[:,1], 'o')
    pyplot.title('data2')

    pyplot.subplot(3,4,6)
    pyplot.plot(pc2[:,0], pc2[:,1], 'o')
    pyplot.title('pc2')

    if not usePcaSvd:
        pyplot.subplot(3,4,7)
        pyplot.plot(pc2white[:,0], pc2white[:,1], 'o')
        pyplot.title('pc2white')

    pyplot.subplot(3,4,8)
    pyplot.plot(recon2[:,0], recon2[:,1], 'o')
    pyplot.plot(pc2whiteRecon[:,0]+.2, pc2whiteRecon[:,1]+.2, 'ro')
    pyplot.title('recon2 and pc2whiteRecon')

    pyplot.subplot(3,4,9)
    pyplot.plot(zca1Manual[:,0], zca1Manual[:,1], 'o')
    pyplot.plot(zca1[:,0]+.05, zca1[:,1]+.05, 'ro')
    pyplot.title('zca1Manual and zca1')

    pyplot.subplot(3,4,10)
    pyplot.plot(zca2Manual[:,0], zca2Manual[:,1], 'o')
    pyplot.plot(zca2[:,0]+.05, zca2[:,1]+.05, 'ro')
    pyplot.title('zca2Manual and zca2')

    pyplot.subplot(3,4,11)
    pyplot.semilogy(pca.var, 'o-')
    pyplot.title('pc1.var')

    pyplot.subplot(3,4,12)
    pyplot.plot(recon2_1dim[:,0], recon2_1dim[:,1], 'o')
    pyplot.title('recon2[1 dim]')



if __name__ == '__main__':
    testPca(PcaClass = PCA)
    raw_input('push enter to exit')
