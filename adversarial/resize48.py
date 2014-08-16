from pylearn2.utils import image
import numpy as np
for i in xrange(900):
    print i
    number = str(i)
    while len(number) != 3:
        number = '0' + number
    img = image.load('video/' + number + '.png')
    out = np.zeros((480, 480, 3))
    for j in range(48):
        for k in xrange(48):
            for l in xrange(3):
                out[j*10:(j+1)*10, k*10:(k+1)*10, l] = img[j,k,0]
    image.save('video_resized/' + number + '.png', out)
