from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.gui.patch_viewer import PatchViewer

dataset = CIFAR10(which_set = 'test')

pv = PatchViewer((10,1),(32,32),is_color=True)

T,y = dataset.get_batch_topo(10, include_labels = True)

for i in xrange(10):
    print dataset.label_names[y[i]]
    pv.add_patch(dataset.adjust_for_viewer(T[i,:,:,:]),rescale=False)

pv.show()
