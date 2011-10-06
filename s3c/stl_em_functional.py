#arg1: model to evaluate
#arg2: batch size
#arg3: # batches

import sys

model_path = sys.argv[1]

from pylearn2.utils import serial

model = serial.load(model_path)

batch_size = int(sys.argv[2])
num_batches = int(sys.argv[3])


