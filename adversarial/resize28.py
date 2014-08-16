from pylearn2.utils import image
import numpy as np
for i in xrange(900):
    print i
    number = str(i)
    while len(number) != 4:
        number = '0' + number
    img = image.load('/Tmp/video/' + number + '.png')
    out = np.zeros((480, 480, 3))
    for j in range(28):
        for k in xrange(28):
            for l in xrange(3):
                out[j*17:(j+1)*17, k*17:(k+1)*17, l] = img[j,k,0]
    image.save('/Tmp/video_resized/' + number + '.png', out)
