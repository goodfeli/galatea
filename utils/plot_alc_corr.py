# Plot correlation between the valid/test alc score and the true
# alc score obtained from the website
#
# Usage: python plot_corr.py <file>
from pylab import *
import sys

def corr_plot(data="alc_vs_vt_sylvester.log"):
    res = loadtxt(data)
    f = figure()
    scatter(res[:,0], res[:,1])
    title("Correlation between train alc and true alc for " + data)
    xlabel("train alc")
    ylabel("true valid alc")

    for i in range(res.shape[0]):
        print res[i,2]
        annotate(str(int(res[i,2])), (res[i,0], res[i,1]))

    savefig(data[:-4])

    return f

if __name__ == "__main__":
    if len(sys.argv) > 1:
        f = corr_plot(data=sys.argv[1])
    else:
        print "Usage: python plot_alc_corr.py <file>"


