#! /usr/bin/env python

'''
Research code

Jason Yosinski

Utilities for learning. Some stuff copied from
https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/utils.py
'''

import ipdb as pdb
import numpy
from numpy import mgrid, array, ones, zeros, linspace, random, reshape

from mayavi import mlab
from mayavi.mlab import points3d, contour3d, plot3d
from tvtk.api import tvtk



#########################
#
# 3D Shapes
#
#########################

cubeEdges = array([[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                   [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
                   [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1]])



def plot3DShapeFromFlattened(blob, blobShape,
                             saveFilename = None, smoothed = False, visSimple = True,
                             plotThresh = 0, figSize = (300,300)):
    '''Plots a flattened 3D shape of size blobShape inside a frame.'''

    if type(blobShape) is list or type(blobShape) is tuple:
        Nx,Ny,Nz = blob.shape
    else:
        Nx,Ny,Nz = blobShape,blobShape,blobShape

    return plot3DShape(reshape(blob, (Nx,Ny,Nz)), saveFilename, smoothed, visSimple,
                       plotThresh, figSize)



def plot3DShape(blob, saveFilename = None, smoothed = False, visSimple = True,
                plotThresh = 0, figSize = (300,300), plotEdges = True, rotAngle = 24):
    '''Plots a 3D shape inside a frame.'''

    Nx,Ny,Nz = blob.shape
    indexX, indexY, indexZ = mgrid[0:Nx,0:Ny,0:Nz]
    edges = (array([Nx,Ny,Nz]) * cubeEdges.T).T
    
    fig = mlab.figure(0, size = figSize)
    mlab.clf(fig)
    fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
    if plotEdges:
        plot3d(edges[0,:], edges[1,:], edges[2,:], color=(.5,.5,.5),
               line_width = 0,
               representation = 'wireframe',
               opacity = 1)

    # Convert from bool to real if necessary
    if type(blob[0]) is numpy.bool_ or blob.dtype == numpy.dtype('bool'):
        blob = blob * 1

    if smoothed:
        # Pad with zeros to close, large negative values to make edges sharp
        bs1,bs2,bs3 = blob.shape
        blob = numpy.pad(blob, 1, 'constant', constant_values = (-1000,))
        contour3d(blob, extent=[0,bs1,0,bs2,0,bs3], contours=[.1], color=(1,1,1))
    else:
        mn = blob.min()
        mx = blob.max()
        idx = (blob > plotThresh).flatten()
        #print '  plot3DShape:', mn, mx, sum(idx)
        if sum(idx) > 0:
            if visSimple:
                pts = points3d(indexX.flatten()[idx] + .5,
                               indexY.flatten()[idx] + .5,
                               indexZ.flatten()[idx] + .5,
                               ones(sum(idx)) * .9,
                               #((blob-mn) / (mx-mn) * .9)[idx],
                               color = (1,1,1),
                               mode = 'cube',
                               scale_factor = 1.0)
            else:
                pts = points3d(indexX.flatten()[idx] + .5,
                               indexY.flatten()[idx] + .5,
                               indexZ.flatten()[idx] + .5,
                               #ones(sum(idx)) * .9,
                               ((blob-mn) / (mx-mn) * .9)[idx],
                               colormap = 'bone',
                               #color = (1,1,1),
                               mode = 'cube',
                               scale_factor = 1.0)
            lut = pts.module_manager.scalar_lut_manager.lut.table.to_array()
            tt = linspace(0, 255, 256)
            lut[:, 0] = tt*0 + 255
            lut[:, 1] = tt*0 + 255
            lut[:, 2] = tt*0 + 255
            lut[:, 3] = tt
            pts.module_manager.scalar_lut_manager.lut.table = lut

    #mlab.view(57.15, 75.55, 50.35, (7.5, 7.5, 7.5)) # nice view
    #mlab.view(24, 74, 33, (5, 5, 5))      # Default older RBM
    mlab.view(rotAngle, 88, 45, (5, 5, 10))      # Good for EF

    mlab.draw()
    
    if saveFilename:
        print saveFilename
        mlab.savefig(saveFilename)




def justPlotBoolArray(blob, figSize = (300,300)):
    '''Plots a 3D boolean array with points where array is True'''

    Nx,Ny,Nz = blob.shape

    indexX, indexY, indexZ = mgrid[0:Nx,0:Ny,0:Nz]
    
    fig = mlab.figure(0, size = figSize)
    mlab.clf(fig)
    fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

    idx = blob
    print idx.sum(), 'points'
    
    if idx.sum() > 0:
        idxFlat = idx.flatten()
        pts = points3d(indexX.flatten()[idxFlat] + .5,
                       indexY.flatten()[idxFlat] + .5,
                       indexZ.flatten()[idxFlat] + .5,
                       ones(sum(idxFlat)) * .9,
                       #((blob-mn) / (mx-mn) * .9)[idx],
                       color = (1,1,1),
                       mode = 'cube',
                       scale_factor = 1.0)
        lut = pts.module_manager.scalar_lut_manager.lut.table.to_array()
        tt = linspace(0, 255, 256)
        lut[:, 0] = tt*0 + 255
        lut[:, 1] = tt*0 + 255
        lut[:, 2] = tt*0 + 255
        lut[:, 3] = tt
        pts.module_manager.scalar_lut_manager.lut.table = lut

    #mlab.view(57.15, 75.55, 50.35, (7.5, 7.5, 7.5)) # nice view
    #mlab.view(24, 74, 33, (5, 5, 5))      # Default older RBM
    mlab.view(24, 88, 45, (5, 5, 10))      # Good for EF

    mlab.draw()
