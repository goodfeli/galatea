'''
_kmeans_simple is a simple implementation of kmeans
It is adjusted from scikits.learn and the original copyright info is as follows.
'''

# Authors: Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Thomas Rueckstiess <ruecksti@in.tum.de>
#          James Bergstra <james.bergstra@umontreal.ca>
#          Jan Schlueter <scikit-learn@jan-schlueter.de>
#          Nelle Varoquaux
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
# License: BSD

import numpy as np
from jiayq.utils import fastop, gemm, mpiutils
import exceptions
from jiayq.utils.timer import Timer

comm = mpiutils.comm
rank = mpiutils.rank
size = mpiutils.size

def euclidean_distances(X, Y=None, Y_norm_squared=None,squared=False):
    if Y is None:
        Y = X
    if Y_norm_squared is None:
        Y_norm_squared = np.sum(Y**2,axis=1)
    X_norm_squared = np.sum(X**2,axis=1)
    distance = gemm.mygemm(-2.0, X, Y.T)
    #distance = np.dot(X,Y.T)*-2.0
    distance += Y_norm_squared
    distance += np.atleast_2d(X_norm_squared).T
    if squared:
        return distance
    else:
        return np.sqrt(distance)

def l1_distance(X,Y=None):
    if Y is None:
        Y = X
    distance = np.empty((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        distance[i] = np.sum(np.abs(Y - X[i]),axis=1)
    return distance

def k_init(X, k, n_local_trials=None, x_squared_norms=None):
    """Init k seeds according to kmeans++

    Parameters
    -----------
    X: array, shape (n_samples, n_features)
        The data to pick seeds for

    k: integer
        The number of seeds to choose

    n_local_trials: integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    x_squared_norms: array, shape (n_samples,), optional
        Squared euclidean norm of each data point. Pass it if you have it at
        hands already to avoid it being recomputed here. Default: None

    Notes
    ------
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, n_features = X.shape

    centers = np.empty((k, n_features))

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(k))

    # Pick first center randomly
    center_id = np.random.randint(n_samples)
    centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    if x_squared_norms is None:
        x_squared_norms = np.sum(X**2,axis=1)
    closest_dist_sq = euclidean_distances(
        np.atleast_2d(centers[0]), X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining k-1 points
    for c in range(1, k):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = np.random.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(), rand_vals)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in xrange(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq
    return centers

###############################################################################
# K-means estimation by EM (expectation maximisation)

def kmeans(X, k, init=None, n_init=1, max_iter=300, verbose=0, tol=1e-4, distance='l2', parallel = True):
    """ K-means clustering algorithm.

    Parameters
    ----------
    X: ndarray
        A M by N array of M observations in N dimensions or a length
        M array of M one-dimensional observations.

        If we run kmeans under MPI, then X in every MPI node is the
        local data points it is responsible for.

    k: int or ndarray
        The number of clusters to form as well as the number of
        centroids to generate.
        if k is an ndarray, then we do kmeans prediction

    max_iter: int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.

    n_init: int, optional, default: 1
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    tol: float, optional
        The relative increment in the results before declaring convergence.

    verbose: boolean, optional
        Verbosity mode

    distance: string, optional
        distance measure to use. default 'l2', can be:
        'l2', 'l1', more to come

    Returns
    -------
    centroid: ndarray
        A k by N array of centroids found at the last iteration of
        k-means.

    label: ndarray
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia: float
        The final value of the inertia criterion

    """
    distance = distance.lower()
    if type(k) is np.ndarray:
        # do kmeans prediction
        return _e_step(X,k,distance)[0]

    if parallel and ((distance != 'l2' and distance != 'euclidean') or init == 'kmeans++'):
        raise NotImplementedError, 'mpi kmeans only available for l2 distances, with random initialization.'
    if parallel:
        mpiutils.rootprint('Parallel K-means in play.')


    #otherwise, do kmeans training
    vdata = np.mean(np.var(X, 0))
    best_inertia = np.infty

    # precompute squared norms of data points
    x_squared_norms = np.sum(X**2,axis=1)
    for it in range(n_init):
        # init
        if init == 'kmeans++':
            centers = k_init(X, k, x_squared_norms=x_squared_norms)
        else:
            centers = X[np.random.randint(X.shape[0],size=k)]
        if parallel:
            # if we have more than 1 nodes, we need to have them agree on centers
            centers_all = comm.gather(centers)
            if rank == 0:
                centers_all = np.vstack(centers_all)
                centers[:] = centers_all[np.random.permutation(centers_all.shape[0])[:k]]
            comm.Bcast(centers, root=0)

        if verbose:
            mpiutils.rootprint('Initialization complete')

        # iterations
        timer = Timer()
        for i in range(max_iter):
            centers_old = centers.copy()
            labels, inertia = _e_step(X, centers, distance,
                                      x_squared_norms=x_squared_norms)
            if parallel:
                # gather inertia
                inertia = comm.allreduce(inertia)
            if verbose:
                mpiutils.rootprint('Attempt %i, Iteration %i, inertia %s, this iteration took %ss.' % (it, i, inertia, timer.lap()) )

            centers = _m_step(X, labels, k, distance, parallel)
            # debug code
            #print 'After update centers: inertia %s' % (_calc_inertia(X, centers, labels))

            # for numerical stability, convergence is only carried out on rank 0
            converged = 0
            if rank == 0:
                if np.sum((centers_old - centers) ** 2) < tol * vdata:
                    if verbose:
                        mpiutils.rootprint('Converged to similar centers at iteration {}'.format(i))
                    converged = 1
            converged = comm.bcast(converged, root=0)
            if converged:
                break

        if inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia
    return best_centers, best_labels, best_inertia

def _m_step(X, z, k, distance, parallel):
    """M step of the K-means EM algorithm

    Computation of cluster centers/means

    Parameters
    ----------
    X: array, shape (n_samples, n_features)

    z: array, shape (n_samples)
        Current assignment

    k: int
        Number of desired clusters

    Returns
    -------
    centers: array, shape (k, n_features)
        The resulting centers
    """
    '''
    #old method - use jiayq.fastop instead.
    dim = X.shape[1]
    centers = np.empty((k, dim))
    for q in range(k):
        center_mask = np.flatnonzero(z == q)
        if len(center_mask) == 0:
            # randomly select a data point for the center
            centers[q] = X[np.random.randint(X.shape[0])]
        else:
            centers[q] = fastop.fastsumm(X, center_mask) / len(center_mask)
            #centers[q] = np.mean(X[center_mask], axis=0)
    print centers
    '''
    if distance == 'l2' or distance == 'euclidean':
        centers, counts = fastop.fastcenters(X, z, k)
        if parallel:
            # we need to gather centers
            sums = centers * counts.reshape(k,1)
            allcounts = counts.copy()
            comm.Allreduce(sums, centers)
            comm.Allreduce(counts, allcounts)
            for i in range(k):
                if allcounts[i] == 0:
                    # produce a new count
                    president = mpiutils.vote()
                    if rank == president:
                        centers[i] = X[np.random.randint(X.shape[0])]
                    comm.Bcast(centers[i],root=president)
                else:
                    centers[i] /= allcounts[i]
        else:
            emptyclusters = np.flatnonzero(counts==0)
            for i in emptyclusters:
                centers[i] = X[np.random.randint(X.shape[0])]
    elif distance == 'l1':
        centers = np.zeros((k, X.shape[1]))
        for i in range(k):
            idx = (z==i)
            if np.any(idx):
                np.median(X[z==i].copy(), axis=0, out=centers[i], overwrite_input=True)
            else:
                centers[i] = X[np.random.randint(X.shape[0])]
    else:
        raise exceptions.NotImplementedError, "Kmeans distance not implemented"
    return centers


def _e_step(x, centers, distance, x_squared_norms=None):
    """E step of the K-means EM algorithm

    Computation of the input-to-cluster assignment

    Parameters
    ----------
    x: array, shape (n_samples, n_features)

    centers: array, shape (k, n_features)
        The cluster centers

    x_squared_norms: array, shape (n_samples,), optional
        Squared euclidean norm of each data point, speeds up computations in
        case of precompute_distances == True. Default: None

    Returns
    -------z: array of shape(n)
        The resulting assignment

    inertia: float
        The value of the inertia criterion with the assignment
    """

    n_samples = x.shape[0]
    minibatch = 1000
    #if x.shape[0] <= minibatch:
    #    distances = euclidean_distances(centers, x, x_squared_norms, squared=True)
    #    minid = np.argmin(distances, axis=0)
    #    inertia = np.sum(distances[minid, range(n_samples)])
    #else:
    #print 'Dataset too large. Using minibatches'
    minid = np.empty(n_samples, dtype=np.int)
    inertia = 0.0
    for start in range(0, n_samples, minibatch):
        end = min(n_samples, start + minibatch)
        if distance == 'l2':
            if x_squared_norms is None:
                x_squared_norms = np.sum(x**2,axis=1)
            distances = euclidean_distances(centers, x[start:end], x_squared_norms[start:end], squared=True)
        elif distance == 'l1':
            distances = l1_distance(centers, x[start:end])
        else:
            raise exceptions.NotImplementedError, "Kmeans distance not implemented"
        minid[start:end] = np.argmin(distances, axis=0)
        inertia += np.sum(distances[minid[start:end], range(end - start)])

    return minid, inertia

'''
def _calc_inertia(x, centers, label, x_squared_norm=None):
    if x_squared_norm is None:
        x_squared_norm = np.sum(x**2,axis=1)
    minibatch = 1000
    n_samples = x.shape[0]
    inertia = 0.0
    for start in range(0, n_samples, minibatch):
        end = min(n_samples, start + minibatch)
        distances = euclidean_distances(centers, x[start:end], x_squared_norm[start:end], squared=True)
        inertia += np.sum(distances[label[start:end], range(end - start)])
    return inertia
'''

def test():
    data = np.vstack((np.random.randn(500,2)+1,\
                      np.random.randn(500,2)-1))
    centers, assignments, score = kmeans(data, 8, n_init=2, max_iter=100, verbose=1, distance='l2')
    print 'score =', score
    print 'centers = \n', centers
    try:
        from matplotlib import pyplot
        pyplot.scatter(data[:,0],data[:,1],c=assignments)
        pyplot.show()
    except Exception:
        print 'cannot show figure. will simply pass'
        pass

if __name__=="__main__":
    test()
