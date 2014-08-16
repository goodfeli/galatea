from pylearn2.utils import image
import numpy as np
for i in xrange(5000):
    print i
    number = str(i)
    while len(number) != 4:
        number = '0' + number
    img = image.load('video/' + number + '.png')
    out = np.zeros((480, 480, 3))
    for j in range(32):
        for k in xrange(32):
            for l in xrange(3):
                out[j*15:(j+1)*15, k*15:(k+1)*15, l] = img[j,k,l]
    image.save('video_resized/' + number + '.png', out)
