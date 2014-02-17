#! /usr/bin/env python

import cPickle as pickle
import numpy
from numpy import array, zeros, reshape, random, vstack, concatenate, copy, dot, minimum, maximum, asarray
import os
import gzip
import ipdb as pdb
import sys
import scipy
import scipy.io

from fileIO import loadFromPklGz, saveToFile
from cache import cached, cached2, cached2jm



def loadCifarData(cifarDirectory):
    ''' Loads the CIFAR-10 dataset.'''

    if cifarDirectory[-1] != '/':
        cifarDirectory += '/'

    fileNames = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4','data_batch_5', 'test_batch']

    dicts = []
    for fileName in fileNames:
        with open(cifarDirectory + fileName, 'rb') as ff:
            dicts.append(pickle.load(ff))

    batchMeta    = dicts[0]
    trainBatches = dicts[1:6]
    testBatch    = dicts[6]

    train_set_x = zeros((50000, 1024*3), dtype=numpy.float32)
    train_set_y = zeros(50000, dtype=numpy.int8)
    for ii, trainBatch in enumerate(trainBatches):
        train_set_x[ii*10000:(ii+1)*10000, :] = trainBatch['data']
        train_set_y[ii*10000:(ii+1)*10000]    = trainBatch['labels']
    test_set_x = array(testBatch['data'], dtype=numpy.float32)
    test_set_y = array(testBatch['labels'], dtype=numpy.int8)

    classNames = batchMeta['label_names']

    # no validation set,
    return [[train_set_x, train_set_y], [array([]), None], [test_set_x, test_set_y]], classNames



def loadCifarDataMonochrome(cifarDirectory):
    ''' Loads the CIFAR-10 dataset but makes it monochrome'''

    datasets, classNames = loadCifarData(cifarDirectory)

    train_set_x_mono = zeros((50000, 1024), dtype=numpy.float32)
    test_set_x_mono = zeros((10000, 1024), dtype=numpy.float32)
    for ii in range(3):
        train_set_x_mono += datasets[0][0][:, ii*1024:(ii+1)*1024]
        test_set_x_mono  += datasets[2][0][:, ii*1024:(ii+1)*1024]
    train_set_x_mono /= 3
    test_set_x_mono /= 3

    # no validation set,
    return [[train_set_x_mono, datasets[0][1]], datasets[1], [test_set_x_mono, datasets[2][1]]], classNames



def loadCifarDataSubsets(cifarDirectory, size, topLeftCoords):
    ''' Loads the CIFAR-10 dataset but chops it up into
    len(topLeftCoords) windows of size SIZE, where this first window
    is anchored at (ii,jj) = topLeftCoords[0], etc...'''

    datasets, classNames = loadCifarData(cifarDirectory)

    print 'Abandoned for now... pick up later, maybe'



def loadAtariData(filename):
    '''Loads Atari Data'''

    data = loadFromPklGz(filename)
    data = data.T   # Make into one example per column
    return data



def loadUpsonData(filename):
    '''Loads Upson Data'''

    data = loadFromPklGz(filename)
    data = data.T   # Make into one example per column
    return data



def loadUpsonData3(filename):
    '''Loads Upson Data from the upson_rovio_3 dataset (with labels)'''

    data,labels,labelStrings = loadFromPklGz(filename)
    data = data.T   # Make into one example per column
    labels = labels.T
    return data, labels, labelStrings



def loadRandomData(filename):
    '''Loads Random Data'''

    data = loadFromPklGz(filename)
    return data



def loadNYU2Data(patchSize, number, rgbColors = 3, depthChannels = 1, seed = None, filename = '../data/nyu_local/nyu_depth_v2_labeled.mat'):
    '''Load the supervised portion of the NYU2 dataset direclty from the
    nyu_depth_v2_labeled.mat file. See:
    http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
    http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat

    params: rgbColors: 0 for no data, 1 for grayscale images, 3 for color images
            depthChannels: 0 for no depth channel, 1 for depth channel
    Note: sum([rgbColors, depthChannels]) must be > 0.

    Note: For color images, data flattens to [ii_r ii_g ii_b      ii+1_r ii+1_g ii+1_b ...]
          For RGBD images, data  flattens to [ii_r ii_g ii_b ii_d ii+1_r ii+1_g ii+1_b ii+1_d ...]

    '''
    
    import h5py

    assert rgbColors in (0, 1, 3)
    assert depthChannels in (0, 1)
    nChannels = rgbColors + depthChannels
    assert nChannels > 0, 'Must load something!'
    
    if not loadNYU2Data._loaded:
        # load data into memory, only once. Takes ~1 min.
        print 'loadNYU2Data: loading', filename, '(could take a while)...',
        sys.stdout.flush()
        ff = h5py.File(filename, 'r')
        loadNYU2Data._depths = array(ff['depths'])
        loadNYU2Data._images = array(ff['images'])
        loadNYU2Data._labels = array(ff['labels'])
        referencedObjects = ff['#refs#']
        loadNYU2Data._names = []
        for ref in ff['names'][0]:
            name = ''.join([str(unichr(x)) for x in referencedObjects[ref]])
            loadNYU2Data._names.append(name)
        loadNYU2Data._scenes = []
        for ref in ff['scenes'][0]:
            name = ''.join([str(unichr(x)) for x in referencedObjects[ref]])
            loadNYU2Data._scenes.append(name)
        loadNYU2Data._sceneTypes = []
        for ref in ff['sceneTypes'][0]:
            name = ''.join([str(unichr(x)) for x in referencedObjects[ref]])
            loadNYU2Data._sceneTypes.append(name)
        loadNYU2Data._loaded = True
        print 'done.'

    rng = random.RandomState(seed)      # if seed is None, this takes its seed from timer

    # loadNYU2Data._images.shape = (1449, 3, 640, 480)  <-- note this is jj,ii
    # loadNYU2Data._depths.shape = (1449, 640, 480)     <-- note this is jj,ii
    # loadNYU2Data._labels.shape = (1449, 640, 480)     <-- note this is jj,ii
    Nimages = loadNYU2Data._images.shape[0]
    maxI = loadNYU2Data._images.shape[3] - patchSize[0]
    maxJ = loadNYU2Data._images.shape[2] - patchSize[1]

    randomSamples = vstack((rng.randint(0, Nimages, number),
                            rng.randint(0, maxI+1, number),
                            rng.randint(0, maxJ+1, number))).T
    singleChannelLength = patchSize[0] * patchSize[1]
    imageMatrix = zeros((number, singleChannelLength * nChannels), dtype = numpy.float32)
    labelMatrix = zeros((number, singleChannelLength), dtype = numpy.uint16)

    rgb2L = array([.299, .587, .114])  # same as PIL

    print 'loadNYU2Data: grabbing', number, 'samples (could take a while)...',
    sys.stdout.flush()

    for count, sample in enumerate(randomSamples):
        idx, ii, jj = sample
        if count % 10000 == 0:
            #print 'loadNYU2Data: %d / %d' % (count, number)
            pass

        if rgbColors > 0:
            # Grab imgRegion
            imgRegion = array(loadNYU2Data._images[idx,:,jj:(jj+patchSize[1]),ii:(ii+patchSize[0])].transpose((2,1,0)),
                              order = 'C', copy = True, dtype = numpy.float32)
            imgRegion /= 255    # normalize to 0-1 range

            if rgbColors == 1:
                # grayscale images, no depth
                imgRegion = dot(reshape(imgRegion, (-1,3)), rgb2L)
                imgRegion = reshape(imgRegion, imgRegion.shape + (1,))

        if depthChannels > 0:
            # Grab depRegion
            depRegion = copy(loadNYU2Data._depths[idx,  jj:(jj+patchSize[1]),ii:(ii+patchSize[0])].T, order='C')
            depRegion = reshape(depRegion, depRegion.shape + (1,))
            # loadNYU2Data._depths.min() = 0.71329951, loadNYU2Data._depths.max() = 9.99547
            depRegion /= 10     # approx normalize to 0-1 range

        if rgbColors == 0:
            concatRegion = depRegion
        elif depthChannels == 0:
            concatRegion = imgRegion
        else:
            concatRegion = concatenate((imgRegion, depRegion), axis = 2)
        imageMatrix[count,:] = concatRegion.flatten()

        labelRegion = copy(loadNYU2Data._labels[idx, jj:(jj+patchSize[1]),ii:(ii+patchSize[0])].T, order='C')
        labelMatrix[count,:] = labelRegion.flatten()

    print 'done.'
    # one example per column
    return imageMatrix.T, labelMatrix.T

# Note: we cache the large matrices by attaching them to the function
# instead of in a class object so that the higher level util.cache
# framework will work
loadNYU2Data._loaded = False
loadNYU2Data._depths = None
loadNYU2Data._images = None
loadNYU2Data._labels = None
loadNYU2Data._names = None
loadNYU2Data._scenes = None
loadNYU2Data._sceneTypes = None



def loadCS294Images(patchSize = (8,8), number = 10000, seed = None, filename = '../data/stanford_cs294a_images.mat'):
    '''Load pre-whitened patches from CS294 dataset directly from the nyu_depth_v2_labeled.mat file.
    See: http://www.stanford.edu/class/cs294a/handouts.html
    '''

    loaded = scipy.io.loadmat(filename)
    images = loaded['IMAGES']

    rng = random.RandomState(seed)      # if seed is None, this takes its seed from timer

    # loadNYU2Data._images.shape = (1449, 3, 640, 480)  <-- note this is jj,ii
    # loadNYU2Data._depths.shape = (1449, 640, 480)     <-- note this is jj,ii
    # loadNYU2Data._labels.shape = (1449, 640, 480)     <-- note this is jj,ii
    Nimages = images.shape[2]
    maxI = images.shape[0] - patchSize[0]
    maxJ = images.shape[1] - patchSize[1]

    randomSamples = vstack((rng.randint(0, Nimages, number),
                            rng.randint(0, maxI+1, number),
                            rng.randint(0, maxJ+1, number))).T
    singleChannelLength = patchSize[0] * patchSize[1]
    imageMatrix = zeros((number, singleChannelLength), dtype = numpy.float32)

    print 'loadCS294Images: grabbing', number, 'samples (could take a while)...',
    sys.stdout.flush()

    for count, sample in enumerate(randomSamples):
        idx, ii, jj = sample
        if count % 10000 == 0:
            #print 'loadNYU2Data: %d / %d' % (count, number)
            pass

        # Grab imgRegion
        imgRegion = array(images[ii:(ii+patchSize[0]), jj:(jj+patchSize[1]), idx],
                          order = 'C', copy = True, dtype = numpy.float32)

        imageMatrix[count,:] = imgRegion.flatten()

    print 'done.'

    # convert to one example per column
    patches = imageMatrix.T

    patches = normalizeCS294(patches)

    return patches



def normalizeCS294(patches):
    # normalize as in sampleIMAGES.m from CS294
    ret = patches.copy()
    ret -= ret.mean(0)
    thresh = ret.std() * 3
    ret = maximum(minimum(ret, thresh), -thresh) / thresh   # scale to -1 to 1
    ret = (ret + 1) * 0.4 + 0.1   #rescale to .1 to .9
    return ret



def approxNormalizeCS294(patches, thresh):
    # normalize approximately as in sampleIMAGES.m from CS294 (but
    # without using values from data, to allow 1. processing test set
    # at the same time as training set, and 2. processing single
    # patches at a time.)
    ret = patches.copy()
    ret -= ret.mean(0)
    ret = maximum(minimum(ret, thresh), -thresh) / thresh   # scale to -1 to 1
    ret = (ret + 1) * 0.4 + 0.1   #rescale to .1 to .9
    return ret



class DataLoader(object):
    pass



def loadNormalizedSaxeData(dataDir, quick, fold):
    trainSegments, trainNames = [], []
    validSegments, validNames = [], []
    testSegments,  testNames  = [], []

    ii = 0
    allSegments = []
    allNames    = []
    while True:
        if quick and ii >= 2:
            print 'WARNING: quick mode, just loading 2 batches'
            break
        try:
            with open(os.path.join(dataDir, 'batch_%02d.normalized.pkl' % ii), 'rb') as ff:
                batchFrames, batchNames = pickle.load(ff)
        except IOError:
            if ii == 0:
                raise
            else:
                break
        allSegments.append(batchFrames)
        allNames.append(batchNames)
        print 'loaded batch %d: %s' % (ii, repr(batchNames))
        ii += 1

    if fold < 0 or fold >= len(allSegments):
        raise Exception('fold is %s but we only have %d batches' % (repr(fold), len(allSegments)))
    allSegments = allSegments[fold:] + allSegments[:fold]
    allNames = allNames[fold:] + allNames[:fold]
    if quick:
        trainSegments = allSegments[0]
        trainNames = allNames[0]
        validSegments = allSegments[1]
        validNames = allNames[1]
        testSegments = None
        testNames = []
    else:
        nBatches = len(allSegments)
        trainSegments = [item for sublist in allSegments[0:nBatches-2] for item in sublist]
        validSegments = [item for sublist in allSegments[nBatches-2:nBatches-1] for item in sublist]
        testSegments  = [item for sublist in allSegments[nBatches-1:nBatches] for item in sublist]
        trainNames = [item for sublist in allNames[0:nBatches-2] for item in sublist]
        validNames = [item for sublist in allNames[nBatches-2:nBatches-1] for item in sublist]
        testNames  = [item for sublist in allNames[nBatches-1:nBatches] for item in sublist]

    return trainSegments, validSegments, testSegments, trainNames, validNames, testNames



class SaxeVideo(DataLoader):
    '''Loads all Saxe video into memory'''

    def __init__(self, dataDir = '../data/saxe', quick = False, seed = 0, fold = 0, dtype = 'float16'):
        self.seed = seed
        self.fold = fold
        self.dtype = dtype    # Data is always read in as float16 (as saved by makeSaxeVideo.py). This is the dtype used for output.
        self.resetRng()

        output = loadNormalizedSaxeData(dataDir, quick, self.fold)
        self.trainSegments, self.validSegments, self.testSegments, self.trainNames, self.validNames, self.testNames = output
        
        self.segments = {'train': self.trainSegments, 'valid': self.validSegments, 'test': self.testSegments}
        self.names = {'train': self.trainNames, 'valid': self.validNames, 'test': self.testNames}


    def resetRng(self):
        self.rng = random.RandomState(self.seed)


    def getRandomBlock(self, length = 60, randomCropShape = None, group = 'train'):
        if group not in ('train', 'valid', 'test'):
            raise Exception('group must be train, valid, or test')
        segments = self.segments[group]
        names    = self.names[group]
        nFrames = array([len(frames) for frames in segments])
        maxStart = nFrames - length   # negative implies the window doesn't fit
        prob = maximum(0.0, maxStart)
        prob /= prob.sum()
        segId = self.rng.choice(len(nFrames), 1, p = prob)[0]
        frameId = self.rng.choice(maxStart[segId])
        #print 'Sampled: %s segment %d, frames %d:%d (%s)' % (group, segId, frameId, frameId+length, names[segId])
        imgShape = segments[segId][frameId:frameId+length, :, :].shape[1:]
        if randomCropShape is None:
            randomCropShape = imgShape
        try:
            startCropII = self.rng.choice(imgShape[0] - randomCropShape[0] + 1)
            startCropJJ = self.rng.choice(imgShape[1] - randomCropShape[1] + 1)
        except ValueError:
            print 'Probably error: Cropping %s img to %s probably failed.' % (repr(imgShape), repr(randomCropShape))
            raise
            
        ret = segments[segId][frameId:frameId+length,
                              startCropII:startCropII+randomCropShape[0],
                              startCropJJ:startCropJJ+randomCropShape[1]]
        return ret

    
    def getRandomBlocks(self, number, length = 60, randomCropShape = None, group = 'train'):
        if number < 1:
            raise Exception('Must get at least one block')
        for ii in range(number):
            if ii == 0:
                block = self.getRandomBlock(length = length, randomCropShape = randomCropShape, group = group)
                # Initialize return matrix
                ret = zeros((number,) + block.shape, dtype = self.dtype)
                ret[ii,:,:,:] = block
            else:
                ret[ii,:,:,:] = self.getRandomBlock(length = length, randomCropShape = randomCropShape, group = group)
        return ret
