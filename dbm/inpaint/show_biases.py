import sys
from pylearn2.utils import get_dataless_dataset
from pylearn2.utils import serial
import numpy as np
from pylearn2.gui.patch_viewer import make_viewer

ignore, model_path = sys.argv

model = serial.load(model_path)
dataset = get_dataless_dataset(model)

biases = model.visible_layer.get_biases()

biases = np.zeros((1,biases.shape[0]))+biases

print 'values: ',(biases.min(), biases.mean(), biases.max())

pv = make_viewer(biases)

pv.show()
