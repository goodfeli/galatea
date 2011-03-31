import numpy
import cPickle
import framework.kmeans as kkk
#from scipi.cluster.vq import vq, kmeans, whiten

def store_PCA_rita(N_COMP, whiten=True):
    '''
    This function stores in memory the dataset preprocessed after
    Local Contrast Normalization + PCA keeping N_COMP first components
    without whitening. Performs the preprocessing for train/valid/test sets
    '''

    prefix = '/data/lisatmp/ift6266h11/rita-pca-lcn/'
    n_drop = 0.
    n_samples = 111808

    # load elements of the PCA computed on training set of rita
    print '... load PCA'
    mu = cPickle.load(open(prefix + 'mu.pkl','r'))
    V = cPickle.load(open(prefix + 'V.pkl','r'))
    S = cPickle.load(open(prefix + 'S.pkl','r'))
    print '... explained variance'
    print S**2 / n_samples
    # components
    if whiten:
        components = numpy.dot(V.T,numpy.diag(1.0/S)*numpy.sqrt(n_samples))[:,:N_COMP]
    else:
        components = V.T[:,:N_COMP]
    print '... we drop the %i first components'%n_drop
    components[:,:n_drop] = 0

    print '... transform valid/test'
    def transform(X, mu, components):
        Xr = X - mu
        Xr = numpy.dot(Xr,components)
        return Xr

    # load data
    # those datasets has already been local normalized
    valid = cPickle.load(open(prefix + 'valid_lcn.pkl','r'))
    test = cPickle.load(open(prefix + 'test_lcn.pkl','r'))

    valid = transform(valid, mu, components)
    test = transform(test, mu, components)

    cPickle.dump(valid, open('./valid_postPCA_' + str(N_COMP) + 'comp.pkl','w'),-1)
    cPickle.dump(test,  open('./test_postPCA_' + str(N_COMP) + 'comp.pkl','w'),-1)

    del valid, testls
	

    # training set has been splited
    train = transform( cPickle.load(open(prefix + 'train_lcn0.pkl','r')), mu, components)
    for i in range(1,8):
        print '... transforming training split %i, our current training set shape is '%i, train.shape
        train = numpy.vstack((train, transform(cPickle.load(open(prefix + 'train_lcn' + str(i) + '.pkl','r')), mu, components)))

    cPickle.dump(train,open('./train_postPCA_' + str(N_COMP) + '.pkl','w'),-1)


if __name__ == '__main__':
    # example: this will store the post-processed train/valid/test/set keeping
    # the 1000 first components
    compNum = 111;
    #store_PCA_rita(compNum)
	
    trainData = cPickle.load(open('train_postPCA_' + str(compNum) + '.pkl','r'))
   # trainData = numpy.load('./train_postPCA_' + str(compNum) + '.pkl')
   # whitened = whiten(trainData)
   # output = kmeans(whitened,9)
   # print 'init Kmeans'
    kmeans=kkk.KMeans(9)
    print 'training kmeans'
    output = kmeans.train(trainData)
    print 'finishedi, dumping output'
    cPickle.dump(output,open('./train_PCA_kmean9.pkl','w'),-1)
