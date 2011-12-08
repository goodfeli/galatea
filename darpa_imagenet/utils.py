import os

def explore_images(path, suffix):
    """ An iterator over all JPG file paths in a directory (recursive)"""

    return ImageIterator(path, suffix)

def count_images(path, suffix):

    count = 0

    for img_path in explore_images(path, suffix):
        #print '\t',img_path
        count += 1

    return count

class ImageIterator:

    def __init__(self, path, suffix = ".JPEG"):
        #print 'making iterator for '+path
        self.path = path
        self.entries = sorted(os.listdir(path))
        self.next_pos = 0
        self.sub_iter = None
        self.suffix = suffix

    def __iter__(self):
        return self

    def next(self):
        #print id(self), 'next'
        if self.sub_iter is not None:
            try:
                return self.sub_iter.next()
            except StopIteration:
                self.sub_iter = None

        if self.next_pos < len(self.entries):
            path = os.path.join(self.path, self.entries[self.next_pos])
            self.next_pos += 1

            if os.path.isdir(path):
                self.sub_iter = ImageIterator(path, self.suffix)
            elif path.endswith(self.suffix):
                return path

            return self.next()
        else:
            raise StopIteration()

