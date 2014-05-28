import sys
from pylearn2.utils import serial
model = serial.load(sys.argv[1])
generator = model.generator
final = generator.mlp.layers[-1]
weights = final.get_weights()
from pylearn2.gui.patch_viewer import make_viewer
make_viewer(weights, is_color=weights.shape[1] % 3 == 0).show()
