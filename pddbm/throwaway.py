from pylearn2.utils import serial

kmeans = serial.load('kmeans.pkl')

mu = kmeans.mu

print (mu.min(),mu.mean(),mu.max())

mu -= .5

mu *= 2

from pylearn2.gui.patch_viewer import make_viewer

pv = make_viewer(mu)

pv.show()
