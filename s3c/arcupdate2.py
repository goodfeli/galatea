import numpy as np
from matplotlib import pyplot as plt

theta = np.arange(0.,np.pi/2.,.01)

alpha = .1

o_theta = alpha * theta

start_x = 1.
start_y = 0.
dest_x = np.cos(theta)
dest_y = np.sin(theta)

blend_x = alpha * dest_x + (1.-alpha) * start_x
blend_y = alpha * dest_y + (1.-alpha) * start_y

tan = blend_y / blend_x

m_theta = np.arctan(tan)

plt.plot(o_theta / m_theta)
#plt.plot(o_theta, m_theta)
plt.show()
