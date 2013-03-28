from PIL import Image
from scipy import io
import numpy as N

image = io.loadmat('fuck_you.mat')['image']

x = Image.fromarray(N.cast['uint8'](255.*image))
x.show()

