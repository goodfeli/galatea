from pylearn2.utils import serial
import sys
import matplotlib.pyplot as plt

d = serial.load(sys.argv[1])
X = d.get_design_matrix()
plt.scatter(X[:,0],X[:,1])
plt.show()
