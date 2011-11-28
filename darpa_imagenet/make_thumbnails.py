from galatea.darpa_imagenet.utils import explore_images
from pylearn2.utils import serial
from pylearn2.utils import image
import numpy as np

input_path = '/data/lisatmp/glorotxa/train'
output_path = '/data/lisatmp/goodfeli/darpa_imagenet'
image_shape = (32,32)

created_subdirs = set([])

for image_path in explore_images(input_path):

    thumbnail_path = image_path.replace(input_path,output_path)
    thumbnail_path = thumbnail_path.replace('.JPEG','.npy')

    thumbnail_subdir = '/'.join(thumbnail_path.split('/')[:-1])

    if thumbnail_subdir not in created_subdirs:
        serial.mkdir(thumbnail_subdir)
        created_subdirs = created_subdirs.union([thumbnail_subdir])

    img = image.load(image_path)
    thumbnail = image.make_letterboxed_thumbnail(img, image_shape)

    np.save(thumbnail_path, thumbnail)


