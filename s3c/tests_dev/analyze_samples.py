import matplotlib.pyplot as plt
import numpy as np

H_params = np.load("H_sampler.npy")
assert len(H_params.shape) == 2
assert H_params.shape[0] == 1
Mu1_params = np.load("Mu1_sampler.npy")
print "Mu1 driven by ",Mu1_params

H = np.load('H_samples.npy')
m,D = H.shape
pos_samples = np.load('pos_samples.npy')
neg_samples = np.load('neg_samples.npy')

final_samples = H * pos_samples + (1-H)*neg_samples

#plt.hexbin(pos_samples[:,0],pos_samples[:,1])
#plt.hexbin(final_samples[:,0],final_samples[:,1])


def make_h_mean_estimate_plot_stable(idx):

    total = 0.

    mean_estimate = np.zeros((m,))

    running_estimate = 0.

    for i in xrange(0,m):

        running_estimate += (H[i,idx]-running_estimate) / float(i+1)

        mean_estimate[i] = running_estimate

    mn = H_params[0,idx]
    std = mean_estimate.std()

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.hold(True)

    ax.plot(mean_estimate)
    ax.plot(np.zeros((m,))+mn)

    ax.set_ylim(mn-3*std,mn+3*std)

def make_h_mean_window_plot_smooth(idx):


    window_size = 100
    window_stride = window_size
    downsampled = []

    for i in xrange(window_size-1,m,window_stride):
        print i

        val = H[i-window_size+1:i,idx].mean()
        downsampled.append(val)

    downsampled = np.asarray(downsampled)


    mn = H_params[0,idx]
    std = downsampled.std()

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.hold(True)

    ax.plot(downsampled)
    ax.plot(np.zeros(downsampled.shape)+mn)



    ax.set_ylim(mn-3*std,mn+3*std)

def make_h_mean_window_plot(idx):

    window_size = 100000
    downsampled = np.zeros((m/window_size,))

    for i in xrange(0,m/window_size):
        downsampled[i] = H[i*window_size:(i+1)*window_size,idx].mean()


    mn = H_params[0,idx]
    std = downsampled.std()

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.hold(True)

    ax.plot(downsampled)
    ax.plot(np.zeros((m/window_size,))+mn)



    ax.set_ylim(mn-3*std,mn+3*std)


def make_h_mean_estimate_plot(idx):
    total = 0.

    mean_estimate = np.zeros((m,))

    for i in xrange(m):
        total += H[i,idx]

        mean_estimate[i] = total / float(i+1)

    mn = H_params[0,idx]
    std = mean_estimate.std()

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.hold(True)

    ax.plot(mean_estimate)
    ax.plot(np.zeros((m,))+mn)



    ax.set_ylim(mn-3*std,mn+3*std)

def make_h_sum_plot(idx):
    total = 0.

    mean_estimate = np.zeros((m,))

    for i in xrange(m):
        total += H[i,idx]

        mean_estimate[i] = total

    mn = H_params[0,idx]
    std = mean_estimate.std()

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.hold(True)

    ax.plot(mean_estimate)
    ax.plot(np.zeros((m,))+mn)



    #ax.set_ylim(mn-3*std,mn+3*std)



#make_h_mean_estimate_plot_stable(idx =0#)
make_h_mean_window_plot_smooth(idx = 0)
#make_h_mean_window_plot(idx=0)

plt.show()
