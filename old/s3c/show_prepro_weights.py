import sys
from pylearn2.utils import serial
from pylearn2.datasets.preprocessing import ZCA
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from pylearn2.gui.patch_viewer import PatchViewer
import numpy as np

path = sys.argv[1]

prepro = serial.load(path)

zca = prepro.items[-1]

assert isinstance(zca,ZCA)

W = zca.P_


assert W.shape[1] % 3 == 0
n = int(np.sqrt(W.shape[1]/3))

d = DenseDesignMatrix(X = W, view_converter = DefaultViewConverter((n,n,3)))

W = d.get_weights_view(W)

pv = PatchViewer(grid_shape = (n*3,n), patch_shape=(n,n), is_color=True)

for i in xrange(n*n*3):
    pv.add_patch( W[i,...], rescale=True)

pv.show()
