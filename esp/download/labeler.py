import os
labeled = set([])

images_path = '/data/lisatmp/goodfeli/esp/word_images'
labels_dir = '/data/lisatmp/goodfeli/esp/word_labels'
from pylearn2.utils import image

while True:
    candidates = os.listdir(images_path)
    for path in candidates:
        word = path.split('.')[0]
        if word in labeled:
            continue
        label_path = labels_dir + '/' + word +'.txt'

        if os.path.exists(label_path):
            continue

        image_path = images_path + '/' + path
        img = image.load(image_path)
        image.show(img)

        f = open(label_path, 'w')

        idx = 1
        print 'Suggested label: ',word
        x = ''
        while x not in ['y', 'n']:
            x = raw_input('ok? ')
            if x == 'y':
                idx += 1
                f.write(word+'\n')

        while True:
            x = raw_input('label %d: ' %idx)
            if x == ' ':
                break
            if x == '.':
                quit()
            f.write(x + '\n')
            idx += 1

        f.close()

