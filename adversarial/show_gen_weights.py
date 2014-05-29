import sys
from pylearn2.utils import serial
model = serial.load(sys.argv[1])
generator = model.generator
i = -1
while True:
    final = generator.mlp.layers[i]
    try:
        weights = final.get_weights()
    except NotImplementedError:
        i -= 1 # skip over SpaceConverter, etc.


from pylearn2.gui.patch_viewer import make_viewer
make_viewer(weights, is_color=weights.shape[1] % 3 == 0).show()
