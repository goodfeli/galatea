import sys
import matplotlib.pyplot as plt
from framework.utils import serial

d = serial.load(sys.argv[1])

dots = d['dots']
acts = d['acts']

assert dots.shape == acts.shape

for i in xrange(dots.shape[1]):
    plt.scatter(dots[:,i],acts[:,i])
    print 'waiting'
    x = raw_input()
    print 'running'
