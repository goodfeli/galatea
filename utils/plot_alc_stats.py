import sys
from pylab import *
from alc_stats import stats, extract_alcs, mobile_corr, mean, median, train4, train2

def plot_exp_alc(data, labels, interactive=False):
    f = figure()
    title("ALC for different experiments")
    xlabel("Experiment number")
    ylabel("ALC")

    for i in range(data.shape[1]):
        plot(data[:,i],  label=labels[i])

    legend()
    savefig('exp_alc')

    if interactive:
        show()

def plot_test_corr(test_alc, data, labels, interactive=False):
    f = figure()
    title("Correlation with test ALC")
    xlabel("test ALC")
    ylabel("ALC")

    for i in range(data.shape[1]):
        plot(test_alc, data[:,i],  label=labels[i])

    legend(loc='upper left')
    savefig('corr_test')

    if interactive:
        show()

def plot_corr(known, known_labels, unknown, unkown_labels, interactive=False):
    f = figure()
    title("Correlation on a 3 values mobile window")
    xlabel("Experiment Offset")
    ylabel("Pearson correlation")

    exp_offsets = range(known.shape[0])

    for i in range(known.shape[1]):
        plot(exp_offsets, known[:,i], '-', label=known_labels[i])

    for i in range(unknown.shape[1]):
        plot(exp_offsets, unknown[:,i], '--', label=unknown_labels[i])

    axis([0, exp_offsets[-1], -1.5, 1.5])
    legend(loc='lower right')
    savefig('mobile_corr')

    if interactive:
        show()

if __name__ == '__main__':
    args    = sys.argv
    options = [ a for a in args if a.find("-") != -1 ]
    args    = [ a for a in args if a.find("-") == -1 ]

    if len(args) >= 2:
        data = loadtxt(sys.argv[1])
    else:
        data = loadtxt('results2.txt')

    if '-i' in options:
        interactive = True
    else:
        interactive = False

    criterias = [mean, median, train4] + [train2(i) for i in range(6)]

    res = stats(data, criterias)
    alcs = extract_alcs(data)

    plot_exp_alc(hstack((alcs, res[:,:3])), ["valid", "test", "train-mean", "train-median", "train"], interactive)

    plot_test_corr(alcs[:,1], vstack((alcs[:,0].T, res[:,:3].T)).T, ["valid", "train-mean", "train-median", "train"], interactive)

    known = zeros((res.shape[0] - 2, 4))
    known_labels = ['train [0 1] vs train [2 3]', 'train [0 2] vs train [1 3]',\
                    'train [0 3] vs train [1 2]', 'train vs valid']

    known[:,0] = mobile_corr(res[:,-6], res[:,-1])
    known[:,1] = mobile_corr(res[:,-5], res[:,-2])
    known[:,2] = mobile_corr(res[:,-4], res[:,-3])
    known[:,3] = mobile_corr(res[:,-7], alcs[:,0])


    unknown = zeros((res.shape[0] - 2, 2))
    unknown_labels = ['valid vs test', 'train vs test']
    unknown[:,0] = mobile_corr(alcs[:,0], alcs[:,1])
    unknown[:,1] = mobile_corr(res[:,-7], alcs[:,1])

    plot_corr(known, known_labels, unknown, unknown_labels, interactive)








