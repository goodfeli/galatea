import os

def explore_images(path):
    """ An iterator over all JPG file paths in a directory (recursive)"""

    return ImageIterator(path)

def count_images(path):

    count = 0

    for img_path in explore_images(path):
        count += 1

    return count

class ImageIterator:

    def __init__(self, path):
        self.path = path
        self.entries = os.listdir(path)
        self.next_pos = 0
        self.sub_iter = None

    def __iter__(self):
        return self

    def next(self):
        if self.sub_iter is not None:
            try:
                return self.sub_iter.next()
            except StopIteration:
                self.sub_iter = None

        if self.next_pos < len(self.entries):
            path = os.path.join(self.path, self.entries[self.next_pos])
            self.next_pos += 1

            if os.path.isdir(path):
                self.sub_iter = ImageIterator(path)
            elif path.endswith('.JPG') or path.endswith('.jpg'):
                return path

            return self.next()

