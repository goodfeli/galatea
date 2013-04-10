import numpy as np

base = '/data/lisa/data/esp_game/ESPGame100k/labels'
import os
paths = sorted(os.listdir(base))
assert len(paths) == 100000

imbase = '/data/lisa/data/esp_game/ESPGame100k/originals'
im_paths = sorted(os.listdir(imbase))
d = {}
for path in im_paths:
    d[path.split('.')[0]] = path


from PIL import Image

idx = 0
for i in xrange(0, len(paths), 128):
    assert len(paths) == 100000
    print 'i',i
    cur_paths = paths[i:i+128]
    if len(cur_paths) != 128:
        assert i + 128 > len(paths)

    batch = np.zeros((3, 200, 200, len(cur_paths)), dtype='float32')
    print 'batch shape: ',batch.shape

    for j in xrange(len(cur_paths)):
        path = cur_paths[j]
        path = path.split('.')[0]
        path = d[path]
        im = Image.open(imbase + '/' + path)
        im.thumbnail((200, 200), Image.ANTIALIAS)
        im = np.array(im).astype('float32')
        im -= im.mean()
        im /= np.sqrt(1e-6 + np.square(im).mean())

        if len(im.shape) != 3:
            assert len(im.shape) == 2
            new = np.zeros((im.shape[0], im.shape[1], 3))
            for k in xrange(3):
                new[:,:,k] = im.copy()
            im = new
        if im.shape[2] == 1:
            new = np.zeros((im.shape[0], im.shape[1], 3))
            for k in xrange(3):
                new[:,:,k] = im.copy()
            im = new
        if im.shape[2] == 4:
            im = im[:, :, 0:3]
        assert im.shape[2] == 3

        im = np.transpose(im, (2, 0, 1))

        row_start = 100 - im.shape[1] / 2
        row_stop = row_start + im.shape[1]
        col_start = 100 - im.shape[2] / 2
        col_stop = col_start + im.shape[2]
        assert col_stop <= 200

        batch[:, row_start:row_stop, col_start:col_stop, j] = im.copy()
    print 'batch shape after: ',batch.shape

    if batch.shape[-1] != 128:
        assert i + 128 > len(paths)
    np.save('/data/lisatmp/goodfeli/hacky_dataset/%d.npy' % idx, batch)

    idx += 1

