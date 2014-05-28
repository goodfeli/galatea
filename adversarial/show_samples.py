from pylearn2.utils import serial
import sys
_, model_path = sys.argv
model = serial.load(model_path)
from pylearn2.gui.patch_viewer import make_viewer
samples = model.generator.sample(100).eval()
print (samples.min(), samples.mean(), samples.max())
viewer = make_viewer(samples * 2.0 - 1.0, is_color = samples.shape[1] % 3 == 0)
viewer.show()
