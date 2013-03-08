from pylearn2.utils import serial
import sys
_, path = sys.argv
import numpy as np

model = serial.load(path)

weights = model.get_weights()

from pylearn2.config import yaml_parse
dataset = yaml_parse.load(model.dataset_yaml_src)

for i in sorted(range(weights.shape[0]), key=lambda x: np.square(weights[:,x].sum())):
    w = weights[:,i]
    idxs = sorted(range(w.shape[0]), key=lambda x: -abs(w[x]))
    print [(w[idx], dataset.words[idx]) for idx in idxs[0:5]]


