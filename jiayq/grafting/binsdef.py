import numpy as np
from exceptions import ValueError

def isPowerOfTwo(x):
    return ( x & (x-1) ==0 )

def rectangularBins(nBinsPerEdge):
    '''
    return all possible rectangular bins for a K*K grid
    '''
    nBins = nBinsPerEdge*nBinsPerEdge
    nMetabins = (nBinsPerEdge*(nBinsPerEdge+1)/2)**2
    metabins = np.empty([nMetabins, nBins], dtype=np.bool)
    metabins[:] = False
    curr_metabin = 0
    temp = np.array(range(nBins), dtype=np.int)
    for i in range(nBinsPerEdge):
        for j in range(i+1,nBinsPerEdge+1):
            for m in range(nBinsPerEdge):
                for n in range(m+1,nBinsPerEdge+1):
                    # the metabin defined by [i:j, m:n]
                    metabins[curr_metabin] = (temp/nBinsPerEdge>=i)&\
                                             (temp/nBinsPerEdge< j)&\
                                             (temp%nBinsPerEdge>=m)&\
                                             (temp%nBinsPerEdge< n)
                    # Test code: output the true/false grid
                    #print i,j,m,n
                    #print metabins[curr_metabin].reshape(nBinsPerEdge, nBinsPerEdge)
                    curr_metabin += 1
    return metabins

def randomBins(nBinsPerEdge, nMetabins, randratio):
    nBins = nBinsPerEdge*nBinsPerEdge
    metabins = np.empty([nMetabins, nBins], dtype=np.bool)
    metabins[:] = False
    for i in range(nMetabins):
        metabins[i] = ( np.random.rand(nBins) < randratio )
        while (not np.any(metabins[i])):
            metabins[i] = ( np.random.rand(nBins) < randratio )
    return metabins

def randomBinsBeta(nBinsPerEdge, nMetabins, alpha, beta):
    '''
    generate random bins, where we impose a conjugate prior over the randratio
    parameter.
    This provides a better cover over different number of active base bins
    '''
    nBins = nBinsPerEdge*nBinsPerEdge
    metabins = np.empty([nMetabins, nBins], dtype=np.bool)
    metabins[:] = False
    for i in range(nMetabins):
        randratio = np.random.beta(alpha,beta)
        metabins[i] = ( np.random.rand(nBins) < randratio )
        while (not np.any(metabins[i])):
            metabins[i] = ( np.random.rand(nBins) < randratio )
    return metabins

def identicalBins(nBinsPerEdge):
    '''
    return the same bins.
    '''
    nBins = nBinsPerEdge*nBinsPerEdge
    nMetabins = nBins
    metabins = np.empty([nMetabins,nBins],dtype=np.bool)
    metabins[:] = False
    for i in range(nBins):
        metabins[i,i] = True
    return metabins

def spmBins(nBinsPerEdge):
    '''
    generate the metabins that are the same as those used in SPM
    '''
    if not isPowerOfTwo(nBinsPerEdge):
        raise ValueError, 'spmBins expects nBinsPerEdge to be power of 2.'
    else:
        metabins = []
        rowid = np.array([range(nBinsPerEdge)]*nBinsPerEdge, dtype=np.int)
        colid = rowid.T.copy()
        levels = int(np.log2(nBinsPerEdge))
        for level in range(levels+1):
            thisrowid = rowid / (2**level)
            thiscolid = colid / (2**level)
            indices = thisrowid * (np.max(thiscolid)+1) + thiscolid
            for index in np.unique(indices):
                metabins.append((indices==index).reshape(indices.size))
    return np.array(metabins)

def printBins(metabins):
    '''
    print the metabins. We will assume that they are square
    '''
    metabins = metabins.astype(int)
    nBinsPerEdge = np.sqrt(metabins.shape[1])
    for i in range(metabins.shape[0]):
        print metabins[i].reshape(nBinsPerEdge,nBinsPerEdge)
