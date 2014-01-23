#! /usr/bin/env python

import os
import time
from matplotlib import pyplot
from numpy import *

from util.pca import PCA
from util.cache import cached, PersistentHasher



class PCAWhiteningDataNormalizer(object):
    '''Uses PCA to white data and optionally project points to the unit ball.'''

    def __init__(self, data, saveDir = None):
        '''Create DataPrepPCA object.
        data: 1 example per column
        saveDir: If set to a string DIR, saves DIR/fracVar.{png,pdf}
        '''

        self.pca = PCA(data.T)

        if saveDir:
            pyplot.figure()
            pyplot.semilogy(self.pca.fracVar, 'o-')
            pyplot.title('Fractional variance in each dimension')
            pyplot.savefig(os.path.join(saveDir, 'fracVar.png'))
            pyplot.savefig(os.path.join(saveDir, 'fracVar.pdf'))
            pyplot.close()


    def raw2normalized(self, data, unitNorm = True):
        '''Projects points from raw space to normalized space.
        returns: (data, extra), where extra may be extra information needed to project back from normalized -> raw
        '''

        data = self.pca.toZca(data.T, epsilon = 1e-6).T
        
        #if saveDir and self.doPlots:
        #    image = Image.fromarray(tile_raster_images(
        #        X = data.T, img_shape = self.imgShape,
        #        tile_shape = (20, 30), tile_spacing=(1,1),
        #        scale_rows_to_unit_interval = True,
        #        scale_colors_together = True))
        #    image.save(os.path.join(saveDir, 'data_white_rescale.png'))
        #    if self.imgIsColor:
        #        image = Image.fromarray(tile_raster_images(
        #            X = data.T, img_shape = self.imgShape,
        #            tile_shape = (20, 30), tile_spacing=(1,1),
        #            scale_rows_to_unit_interval = True,
        #            scale_colors_together = False))
        #        image.save(os.path.join(saveDir, 'data_white_rescale_indiv.png'))

        #if saveDir:
        #    pil_imagesc(cov(data),
        #                saveto = os.path.join(saveDir, 'dataCov_1prenorm.png'))
        if unitNorm:
            # Project each patch to the unit ball
            patchNorms = sqrt(sum(data**2, 0) + (1e-8))
            data = data / patchNorms
            extra = {'patchNorms': patchNorms}
        else:
            extra = {}

        #if saveDir:
        #    pil_imagesc(cov(data),
        #                saveto = os.path.join(saveDir, 'dataCov_2postnorm.png'))

        return data, extra


    def normalized2raw(self, data, extra = None):
        '''Projects points from raw space to normalized space.'''

        return self.pca.fromZca(data.T, epsilon = 1e-6).T


    def __hash__(self):
        hasher = PersistentHasher()
        hasher.update('PCAWhiteningDataNormalizer')
        hasher.update(self.pca)
        return int(hasher.hexdigest(), 16)


    def __cmp__(self, other):
        return self.__hash__() - other.__hash__()



def printDataStats(data, prefix = ''):
    print prefix + ('Current data: %d data points of dimension %d.' % (data.shape[1], data.shape[0])),
    print 'Min %f, Mean %f, Max %f' % (data.min(), data.mean(), data.max())



def main():
    pass



if __name__ == '__main__':
    main()
