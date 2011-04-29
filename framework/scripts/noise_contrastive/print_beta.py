from framework.utils import serial
import sys

for arg in sys.argv[1:]:
    model = serial.load(arg)
    print arg
    print model.beta.get_value()
