from pylearn2.utils import serial

stl10 = serial.load('/data/lisa/data/stl10/stl10_32x32/train.pkl')

batch = stl10.X[24:25,:]

img = stl10.view_converter.design_mat_to_topo_view(batch)[0,...] / 127.5

from pylearn2.gui.patch_viewer import PatchViewer

pv = PatchViewer((27,27),(6,6),pad=(1,1),is_color=True)


pipeline = serial.load('/data/lisa/data/stl10/stl10_patches/preprocessor.pkl')
del pipeline.items[0]
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter

for col in xrange(27):
    for row in xrange(27):
        patch = img[row:row+6,col:col+6]

        d = DenseDesignMatrix( topo_view = patch.reshape(1,6,6,3), view_converter = DefaultViewConverter((6,6,3)) )

        d.apply_preprocessor(pipeline)

        pv.add_patch(d.get_topological_view()[0,...], rescale = True)
pv.show()


