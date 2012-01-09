from pylearn2.utils import serial

stl10 = serial.load('/data/lisa/data/stl10/stl10_32x32/train.pkl')

batch = stl10.X[24:25,:]

img = stl10.view_converter.design_mat_to_topo_view(batch)[0,...] / 127.5

from pylearn2.gui.patch_viewer import PatchViewer

pv = PatchViewer((27,27),(6,6),pad=(1,1),is_color=True)

for row in xrange(27):
    for col in xrange(27):
        pv.add_patch(img[row:row+6,col:col+6], rescale = False)
pv.show()

pv.save('patches.png')

