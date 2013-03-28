import sys
ignore, filepath = sys.argv
import numpy as np

X = np.load(filepath)
std = X.std(axis=0)
print (std.min(), std.mean(), std.max())
