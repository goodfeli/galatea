import numpy as np
from matplotlib import pyplot

decay = 1./1.000015
x = np.arange(1,300*600).astype('float32')
y = np.cast['float32'](.001) * decay ** x

y2 = [ .001 ]
for elem in y[1:]:
    y2.append(np.cast['float32'](y2[-1] * decay))

y2 = np.asarray(y2)

print np.abs(y-y2).max()

assert np.allclose(y, y2)

pyplot.plot(x,y)
pyplot.hold(True)
pyplot.plot(x,y2)
pyplot.show()

