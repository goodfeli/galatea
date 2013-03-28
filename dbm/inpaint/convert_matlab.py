import sys
from pylearn2.utils import serial
from galatea.dbm.inpaint.super_dbm import load_matlab_dbm

_, src, dst = sys.argv

# Note: this only makes sense after the model has been set up
# for joint training; a DBM is not a stack of RBMs
dbm = load_matlab_dbm(src)
serial.save(dst, dbm)
