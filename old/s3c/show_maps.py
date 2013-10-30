import numpy as np
from pylearn2.gui.patch_viewer import PatchViewer
import sys

ignore, path = sys.argv

T = np.load(path)

m, r, c, n = T.shape

s = int(np.sqrt(n))

while s*s < n:
    s += 1

pv = PatchViewer((s,s),(r,c),is_color=False)

for i in xrange(m):
    for j in xrange(n):
        patch = T[i,:,:,j] * 2.0 - 1.0
        pv.add_patch(patch, rescale = False)

    pv.show()

    x = raw_input('waiting..')
