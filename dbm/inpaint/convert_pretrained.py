from pylearn2.utils import serial
from galatea.dbm.inpaint.super_dbm import load_matlab_dbm

# I let Russ's matlab code stitch the two DBMs together, so I don't
# need to make sure I get that right
src = '../dump/stitched.mat'
dst = 'russ/pretrained.pkl'


# Note: this only makes sense after the model has been set up
# for joint training; a DBM is not a stack of RBMs
dbm = load_matlab_dbm(src)
serial.save(dst, dbm)
