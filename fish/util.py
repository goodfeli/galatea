#! /usr/bin/env python

import numpy
from PIL import Image



def imagesc(arr, shape = None, epsilon = 1e-8, saveto = None, cmap = None, show = True, scaleto = 200, peg0 = False):
    '''Like imagesc for Octave/Matlab, but using PIL.'''

    if shape is None and len(arr.shape) == 1:
        print 'If providing imagesc with a column or row vector, must specify shape manually.'
        return
    if shape is None:
        shape = arr.shape
    imarray = numpy.array(arr, dtype = numpy.float32, copy=True)
    if peg0:
        imarray /= 2 * (max(-imarray.min(), imarray.max()) + epsilon)
        imarray += .5
    else:
        imarray -= imarray.min()
        imarray /= (imarray.max() + epsilon)
    imarray = numpy.reshape(imarray, shape)
    if len(shape) > 2:
        # color image given
        arr = numpy.array(imarray * 255, dtype='uint8')
        image = Image.fromarray(arr)
    else:
        # grayscale image given
        if cmap:
            thecmap = cm.get_cmap(cmap)
            imarray = thecmap(imarray)[:,:,:3]  # chop off alpha channel
            imarray = numpy.array(imarray * 255, dtype='uint8')
            image = Image.fromarray(imarray)
        else:
            image = Image.fromarray(imarray * 255).convert('L')

    if saveto:
        image.save(saveto)
    if show:
        factor = max([scaleto / x for x in image.size] + [1])
        imshow  = image.resize([x * factor for x in image.size])
        imshow.show()
    return image

