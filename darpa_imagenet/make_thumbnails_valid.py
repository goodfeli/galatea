from galatea.darpa_imagenet.utils import explore_images
from pylearn2.utils import serial
from pylearn2.utils import image
import numpy as np
import os
import time

input_path = '/data/lisatmp/glorotxa/val'
output_path = '/data/lisatmp/goodfeli/darpa_imagenet_valid'
image_shape = (32,32)

created_subdirs = set([])

for image_path in explore_images(input_path,'.JPEG'):

    thumbnail_path = image_path.replace(input_path,output_path)
    thumbnail_path = thumbnail_path.replace('.JPEG','.npy')

    t1 = time.time()
    e =  os.path.exists(thumbnail_path)
    t2 = time.time()
    print t2-t1

    if e:
        continue

    thumbnail_subdir = '/'.join(thumbnail_path.split('/')[:-1])

    if thumbnail_subdir not in created_subdirs:
        serial.mkdir(thumbnail_subdir)
        created_subdirs = created_subdirs.union([thumbnail_subdir])

    try:
        t1 = time.time()
        img = image.load(image_path)
        t2 = time.time()
    except Exception, e:
        print "Encountered a problem: "+str(e)
        img = None

    if img is not None:
        assert len(img.shape) == 3
        thumbnail = image.make_letterboxed_thumbnail(img, image_shape)
        t3 = time.time()

        np.save(thumbnail_path, thumbnail)
        t4 = time.time()

        print (t2-t1,t3-t2,t4-t3)


