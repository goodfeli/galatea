#! /usr/bin/env python

'''
Research code

Jason Yosinski

Utilities for learning. Some stuff copied from
https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/utils.py
'''

import ipdb as pdb
import numpy
from numpy import sqrt, ceil
from matplotlib import cm
from matplotlib.pyplot import hist, title
from numpy import mgrid, array, ones, zeros, linspace, random, reshape, prod
from PIL import Image



#########################
#
# Images
#
#########################

def scale_to_unit_interval(arr,eps=1e-8):
    ''' Scales all values in the arrray arr to be between 0 and 1 '''
    arr = arr.copy()
    arr -= arr.min()
    arr *= 1.0 / (arr.max()+eps)
    return arr


def scale_all_rows_to_unit_interval(arr,eps=1e-8):
    ''' Scales each row in the 2D array arr to be between 0 and 1 '''
    assert(len(arr.shape) == 2)
    arr = arr.copy()
    arr = (arr.T - arr.min(axis=1)).T
    arr = (arr.T / (arr.max(axis=1)+eps)).T
    return arr


def scale_some_rows_to_unit_interval(arr, rowIdx, eps = 1e-8):
    ''' Scales rows given by rowIdx in the 2D array arr to be between 0 and 1'''
    assert(len(arr.shape) == 2)
    arr = arr.copy()
    for ii in rowIdx:
        arr[ii,:] -= arr[ii,:].min()
        arr[ii,:] /= (arr[ii,:].max() + 1e-8)
    return arr


def scale_rows_together_to_unit_interval(arr, rowIdx, eps = 1e-8, anchor0 = True):
    ''' Scales rows given by rowIdx in the 2D array arr to be between
    0 and 1. If anchor0 is True, fix an input value of 0 to be output
    exactly in the middle of the range (.5)'''
    
    assert(len(arr.shape) == 2)
    arr = arr.copy()

    if not rowIdx:
        print 'WARNING: blank rowIdx. Doing nothing.'
        return

    usemin = arr[rowIdx,:].min()
    usemax = arr[rowIdx,:].max()
    if anchor0:
        maxabs = max(usemax, -usemin, 1e-8)
        for ii in rowIdx:
            arr[ii,:] /= maxabs
            arr[ii,:] += .5
    else:
        for ii in rowIdx:
            arr[ii,:] -= usemin
            arr[ii,:] /= max(usemax - usemin, 1e-8)
    return arr



def tile_raster_images_tensor(tensor, tile_spacing = (0,0),
                              scale_rows_to_unit_interval = True, scale_colors_together = False,
                              output_pixel_vals = True, hilights = None, onlyHilights = False):
    tile_shape = tensor.shape[0:2]
    img_shape  = tensor.shape[2:4]
    reshaped = reshape(tensor, (prod(tile_shape), prod(img_shape)))
    return tile_raster_images(reshaped, img_shape, tile_shape, tile_spacing,
                              scale_rows_to_unit_interval, scale_colors_together,
                              output_pixel_vals, hilights, onlyHilights)



def tile_raster_images(X, img_shape, tile_shape, tile_spacing = (0,0),
                       scale_rows_to_unit_interval = True, scale_colors_together = False,
                       output_pixel_vals = True, hilights = None, onlyHilights = False):
    '''
    Transform an array with one flattened image per row, into an array in 
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images, 
    and also columns of matrices for transforming those rows 
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can 
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)
    
    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not

    :param hilights: list of hilights to apply to each tile. 


    :returns: array suitable for viewing as an image.  
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    '''

    assert(len(img_shape) == 2 or (len(img_shape) == 3 and img_shape[2] == 3)) # grayscale or RGB color
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    isColor = len(img_shape) == 3

    if hilights is not None:
        assert hilights.shape[0] == X.shape[0]
        if len(hilights.shape) == 1:
            hilightIsColor = False
        elif hilights.shape[1] == 3:
            hilightIsColor = True
        else:
            raise Exception('expected hilights.shape to be (X,) or (X,3), but it is %s' % hilights.shape)

        if not isColor and hilightIsColor:
            # promote data to be color
            promotedX = numpy.repeat(X, 3, axis=1)
            promotedImgShape = (img_shape[0], img_shape[1], 3)
            return tile_raster_images(promotedX, promotedImgShape, tile_shape, tile_spacing,
                                      scale_rows_to_unit_interval, scale_colors_together,
                                      output_pixel_vals, hilights, onlyHilights)
        if isColor and not hilightIsColor:
            # promote hilight to be color
            promotedHilight = tile(numpy.atleast_2d(hilights).T, (1,3))
            return tile_raster_images(X, img_shape, tile_shape, tile_spacing,
                                      scale_rows_to_unit_interval, scale_colors_together,
                                      output_pixel_vals, promotedHilight, onlyHilights)

        # Now this must be true
        assert isColor == hilightIsColor

    #pdb.set_trace()

    # The expression below can be re-written in a more C style as 
    # follows : 
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp for ishp, tshp, tsp 
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isColor:
        # massage to expected tuple form
        #assert(X.shape[1] % 3 == 0)
        #nPerChannel = X.shape[1] / 3

        if output_pixel_vals:
            dt = 'uint8'
        else:
            dt = X.dtype
        if str(dt) not in ('uint8', 'float32'):
            raise Exception('color only works for uint8 or float32 dtype, not %s' % dt)

        if scale_rows_to_unit_interval and scale_colors_together:
            X = scale_all_rows_to_unit_interval(X)

        X = X.reshape(X.shape[0], img_shape[0], img_shape[1], img_shape[2])
        X = (X[:,:,:,0].reshape(X.shape[0], img_shape[0] * img_shape[1]),
             X[:,:,:,1].reshape(X.shape[0], img_shape[0] * img_shape[1]),
             X[:,:,:,2].reshape(X.shape[0], img_shape[0] * img_shape[1]),
             None, #numpy.ones(X[:,0:nPerChannel].shape, dtype=dt), # hardcode complete opacity
             )
        if hilights is not None:
            hilights = (hilights[:,0], hilights[:,1], hilights[:,2], None)


    if isinstance(X, tuple):
        assert len(X) == 4
        if hilights is None: hilights = (None, None, None, None)
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            #out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
            out_array = numpy.ones((out_shape[0], out_shape[1], 4), dtype='uint8') * 51
        else:
            #out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)
            out_array = numpy.ones((out_shape[0], out_shape[1], 4), dtype=X.dtype) * .2

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [51,51,51,255]
        else:
            channel_defaults = [.2,.2,.2,1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct 
                # dtype (unless it's the alpha channel)
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                #if i == 3:
                #    # alpha channel
                #    out_array[:,:,i] = numpy.ones(out_shape, dtype=dt)+channel_defaults[i]
                #    pdb.set_trace()
                #else:
                out_array[:,:,i] = numpy.zeros(out_shape, dtype=dt)+channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it 
                # in the output
                doScaleRows = scale_rows_to_unit_interval
                if isColor and scale_colors_together:
                    # already scaled whole rows
                    doScaleRows = False
                out_array[:,:,i] = tile_raster_images(X[i], img_shape[0:2], tile_shape,
                                                      tile_spacing, doScaleRows, False,
                                                      output_pixel_vals, hilights[i], onlyHilights)
        return out_array

    else:
        # if we are dealing with only one channel 
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
            #out_array = numpy.zeros(out_shape, dtype=dt)
            out_array = numpy.ones(out_shape, dtype=dt) * 51
        else:
            out_array = numpy.ones(out_shape, dtype=dt) * .2


        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval and not (isColor and scale_colors_together):
                        # if we should scale values to be between 0 and 1 
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the 
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    # Set slightly bigger region to hilight color
                    if hilights is not None:
                        out_array[
                            tile_row * (H+Hs):tile_row*(H+Hs)+H+Hs,
                            tile_col * (W+Ws):tile_col*(W+Ws)+W+Ws
                            ] \
                            = hilights[tile_row * tile_shape[1] + tile_col] * c
                    if not onlyHilights:
                        out_array[
                            tile_row * (H+Hs)+Hs/2:tile_row*(H+Hs)+H+Hs/2,
                            tile_col * (W+Ws)+Ws/2:tile_col*(W+Ws)+W+Ws/2
                            ] \
                            = this_img * c
        return out_array



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
    imarray = reshape(imarray, shape)
    if len(shape) > 2:
        # color image given
        arr = array(imarray * 255, dtype='uint8')
        image = Image.fromarray(arr)
    else:
        # grayscale image given
        if cmap:
            thecmap = cm.get_cmap(cmap)
            imarray = thecmap(imarray)[:,:,:3]  # chop off alpha channel
            imarray = array(imarray * 255, dtype='uint8')
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



pil_imagesc = imagesc



#########################
#
# General
#
#########################

def looser(ax, frac = .05, semilogx = False, semilogy = False, loglog = False):
    '''Returns a loosened version of the axies specified in ax'''

    ax = list(ax)

    if loglog:
        semilogx = True
        semilogy = True

    if semilogx:
        ax[0] = log(ax[0])
        ax[1] = log(ax[1])
    if semilogy:
        ax[2] = log(ax[2])
        ax[3] = log(ax[3])
        
    ax = [(1+frac) * ax[0] - frac * ax[1], (1+frac) * ax[1] - frac * ax[0],
          (1+frac) * ax[2] - frac * ax[3], (1+frac) * ax[3] - frac * ax[2]]

    if semilogx:
        ax[0] = exp(ax[0])
        ax[1] = exp(ax[1])
    if semilogy:
        ax[2] = exp(ax[2])
        ax[3] = exp(ax[3])
    
    return tuple(ax)



def histPlus(data):
    '''Hist with reasonable default options and magnitude in title. Automatically flattens data before hist.'''

    flat = data.flatten()
    sumsquares = (flat**2).sum()
    mag = sqrt(sumsquares)
    rms = sqrt(sumsquares / len(flat))

    hist(flat, bins = 20, normed = True)
    #title('rms: %g, mag: %g' % (rms, mag), fontdict = {'fontsize': 'small'})
    title('rms: %g, mag: %g' % (rms, mag))



def tile_shape(number_of_tiles, ratio = 1.0, mult = None):
    '''Returns a good tile shape to fit all of the given tiles.
    Ratio is the approximate ncols / nrows'''

    # Method 1
    #rows = int(ceil(sqrt(number_of_tiles)))
    #cols = rows

    # Method 2
    ratio = float(ratio)
    if mult is None:
        cols = int(sqrt(number_of_tiles * ratio)) + 1
        rows = int(ceil(number_of_tiles/float(cols)))
    else:
        rows,cols = tile_shape(number_of_tiles/mult, ratio/mult)
        cols *= mult
        assert cols % mult == 0

    assert rows*cols >= number_of_tiles
    return (rows, cols)
    
