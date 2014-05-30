import sys
from pylearn2.gui.patch_viewer import make_viewer
from pylearn2.utils import serial
model = serial.load(sys.argv[1])
generator = model.generator

final = generator.mlp.layers[-1]
success = False
try:
    topo = final.get_weights_topo()
    success = True
except Exception:
    pass

if success:
    make_viewer(topo).show()
    quit()

i = -1
success = False
while not success:
    final = generator.mlp.layers[i]
    try:
        weights = final.get_weights()
        success = True
    except NotImplementedError:
        i -= 1 # skip over SpaceConverter, etc.


make_viewer(weights, is_color=weights.shape[1] % 3 == 0).show()
