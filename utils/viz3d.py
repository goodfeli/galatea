"""Simple visualization script for 3D features."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def do_3d_scatter(x, y, z, figno=None, title=None):
    """Just generate a 3D scatterplot figure and optionally give it a title."""
    fig = plt.figure(figno)
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.suptitle(title)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scatterplot 3D features '
                                                 'from textfiles, one figure '
                                                 'window per file')
    parser.add_argument('file', action='store',
                        type=argparse.FileType('r'), nargs='+',
                        help='Files to display (loaded by numpy.loadtxt)')
    args = parser.parse_args()
    for i, fn in enumerate(args.file):
        data = np.loadtxt(fn)
        x, y, z = data.T
        do_3d_scatter(x, y, z, figno=(i + 1),
                      title='features from "%s"' % fn.name)
    plt.show()
