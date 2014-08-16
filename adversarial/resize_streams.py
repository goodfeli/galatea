from pylearn2.utils import image
import numpy as np
for i in xrange(5000):
    print i
    number = str(i)
    while len(number) != 4:
        number = '0' + number
    img = image.load('/Tmp/video/' + number + '.png')
    out = np.zeros((454, 454, 3))
    for ofs_r in [0, 1]:
        for ofs_c in [0, 1]:
            out[ofs_r::2, ofs_c::2, :] = img
    image.save('/Tmp/video_resized/' + number + '.png', out)
