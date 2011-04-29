import sys
from framework.utils import serial
import matplotlib.pyplot as plt

model = serial.load(sys.argv[1])

W = model.W.get_value()


plt.scatter(W[0,:],W[1,:])
plt.show()
