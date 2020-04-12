'''
File: meanshit_tf.py
Description: I grabbed the MeanShift algorithm from sklearn and implemented a
             tensorflow version because the sklearn version is too slow.
Author: Ronald Kemker

'''

import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, check_random_state
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.cluster import  get_bin_seeds, estimate_bandwidth
import tensorflow as tf

class MeanShift(BaseEstimator, ClusterMixin):
    """Mean shift clustering using a flat kernel.
    Mean shift clustering aims to discover "blobs" in a smooth density of
    samples. It is a centroid-based algorithm, which works by updating
    candidates for centroids to be the mean of the points within a given
    region. These candidates are then filtered in a post-processing stage to
    eliminate near-duplicates to form the final set of centroids.
    Seeding is performed using a binning technique for scalability.
    Read more in the :ref:`User Guide <mean_shift>`.
    Parameters
    ----------
    bandwidth : float, optional
        Bandwidth used in the RBF kernel.
        If not given, the bandwidth is estimated using
        sklearn.cluster.estimate_bandwidth; see the documentation for that
        function for hints on scalability (see also the Notes, below).
    seeds : array, shape=[n_samples, n_features], optional
        Seeds used to initialize kernels. If not set,
        the seeds are calculated by clustering.get_bin_seeds
        with bandwidth as the grid size and default values for
        other parameters.
    bin_seeding : boolean, optional
        If true, initial kernel locations are not locations of all
        points, but rather the location of the discretized version of
        points, where points are binned onto a grid whose coarseness
        corresponds to the bandwidth. Setting this option to True will speed
        up the algorithm because fewer seeds will be initialized.
        default value: False
        Ignored if seeds argument is not None.
    min_bin_freq : int, optional
       To speed up the algorithm, accept only those bins with at least
       min_bin_freq points as seeds. If not defined, set to 1.
    cluster_all : boolean, default True
        If true, then all points are clustered, even those orphans that are
        not within any kernel. Orphans are assigned to the nearest kernel.
        If false, then orphans are given cluster label -1.
    batch_size : int
        The mini-batch size for assigning labels to each sample.
    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers.
    labels_ :
        Labels of each point.
    Notes
    -----
    Scalability:
    Because this implementation uses a flat kernel and
    a Ball Tree to look up members of each kernel, the complexity will tend
    towards O(T*n*log(n)) in lower dimensions, with n the number of samples
    and T the number of points. In higher dimensions the complexity will
    tend towards O(T*n^2).
    Scalability can be boosted by using fewer seeds, for example by using
    a higher value of min_bin_freq in the get_bin_seeds function.
    Note that the estimate_bandwidth function is much less scalable than the
    mean shift algorithm and will be the bottleneck if it is used.
    References
    ----------
    Dorin Comaniciu and Peter Meer, "Mean Shift: A robust approach toward
    feature space analysis". IEEE Transactions on Pattern Analysis and
    Machine Intelligence. 2002. pp. 603-619.
    """
    def __init__(self, bandwidth=None, seeds=None, bin_seeding=False,
                 min_bin_freq=1, cluster_all=True, batch_size = 128):
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.bin_seeding = bin_seeding
        self.cluster_all = cluster_all
        self.min_bin_freq = min_bin_freq
        self.batch_size = batch_size

    def fit(self, X, y=None):
        """Perform clustering.
        Parameters
        -----------
        X : array-like, shape=[n_samples, n_features]
            Samples to cluster.
        y : Ignored
        """
        X = check_array(X)
        self.cluster_centers_, self.labels_ = \
            mean_shift(X, bandwidth=self.bandwidth, seeds=self.seeds,
                       min_bin_freq=self.min_bin_freq,
                       bin_seeding=self.bin_seeding,
                       cluster_all=self.cluster_all,
                       batch_size = self.batch_size)
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[n_samples, n_features]
            New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, "cluster_centers_")

        return pairwise_distances_argmin(X, self.cluster_centers_)

def mean_shift(X, bandwidth=None, seeds=None, bin_seeding=False,
               min_bin_freq=1, cluster_all=True, max_iter=300, batch_size=128):
    """Perform mean shift clustering of data using a flat kernel.
    Read more in the :ref:`User Guide <mean_shift>`.
    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
        Input data.
    bandwidth : float, optional
        Kernel bandwidth.
        If bandwidth is not given, it is determined using a heuristic based on
        the median of all pairwise distances. This will take quadratic time in
        the number of samples. The sklearn.cluster.estimate_bandwidth function
        can be used to do this more efficiently.
    seeds : array-like, shape=[n_seeds, n_features] or None
        Point used as initial kernel locations. If None and bin_seeding=False,
        each data point is used as a seed. If None and bin_seeding=True,
        see bin_seeding.
    bin_seeding : boolean, default=False
        If true, initial kernel locations are not locations of all
        points, but rather the location of the discretized version of
        points, where points are binned onto a grid whose coarseness
        corresponds to the bandwidth. Setting this option to True will speed
        up the algorithm because fewer seeds will be initialized.
        Ignored if seeds argument is not None.
    min_bin_freq : int, default=1
       To speed up the algorithm, accept only those bins with at least
       min_bin_freq points as seeds.
    cluster_all : boolean, default True
        If true, then all points are clustered, even those orphans that are
        not within any kernel. Orphans are assigned to the nearest kernel.
        If false, then orphans are given cluster label -1.
    max_iter : int, default 300
        Maximum number of iterations, per seed point before the clustering
        operation terminates (for that seed point), if has not converged yet.

    Returns
    -------
    cluster_centers : array, shape=[n_clusters, n_features]
        Coordinates of cluster centers.
    labels : array, shape=[n_samples]
        Cluster labels for each point.
    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_mean_shift.py
    <sphx_glr_auto_examples_cluster_plot_mean_shift.py>`.
    """

    if bandwidth is None:
        bandwidth = estimate_bandwidth(X)
    elif bandwidth <= 0:
        raise ValueError("bandwidth needs to be greater than zero or None,\
            got %f" % bandwidth)
    if seeds is None:
        if bin_seeding:
            seeds = get_bin_seeds(X, bandwidth, min_bin_freq)
        else:
            seeds = X
    n_samples, n_features = X.shape
    center_intensity_dict = {}

    # execute iterations on all seeds
    my_mean_list = []
    points_within_list = []
    with tf.Graph().as_default():
        X_t = tf.placeholder(tf.float32, shape=(n_features, ), name='input')
        X_placeholder_t = tf.placeholder(tf.float32, shape=(X.shape), name='data')
        X_var_t = tf.Variable(X_placeholder_t)        
        dist_t = tf.sqrt(tf.reduce_sum(tf.square(X_t-X_var_t) , axis=1))
        
        idx_t = tf.less(dist_t , bandwidth)
        points_within_t = tf.boolean_mask(X_var_t, idx_t)
        mean_t = tf.reduce_mean(points_within_t , 0)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init, {X_placeholder_t:X})
            for i, seed in enumerate(seeds): 
                print("\rSeeding: %d/%d" %(i+1, len(seeds)),end="")
                stop_thresh = 1e-3 * bandwidth  # when mean has converged
                completed_iterations = 0
                while True:
                # Find mean of points within bandwidth
                    points_within, my_mean = sess.run([points_within_t, mean_t] , feed_dict={X_t : seed}) 
                    if len(points_within) == 0:
                        break  # Depending on seeding strategy this condition may occur
                    my_old_mean = my_mean  # save the old mean
                    # If converged or at max_iter, adds the cluster
                    if (np.linalg.norm(my_mean - my_old_mean) < stop_thresh or
                            completed_iterations == max_iter):
                        my_mean_list.append(tuple(my_mean))
                        points_within_list.append(len(points_within))
                        break
                    completed_iterations += 1
    print("")
    
    # copy results in a dictionary
    for i in range(len(my_mean_list)):
        if points_within_list[i] is not None:
            center_intensity_dict[my_mean_list[i]] = points_within_list[i]
            
    if not center_intensity_dict:
        # nothing near seeds
        raise ValueError("No point was within bandwidth=%f of any seed."
                         " Try a different seeding strategy \
                         or increase the bandwidth."
                         % bandwidth)

    # POST PROCESSING: remove near duplicate points
    # If the distance between two kernels is less than the bandwidth,
    # then we have to remove one because it is a duplicate. Remove the
    # one with fewer points.
    sorted_by_intensity = sorted(center_intensity_dict.items(),
                                 key=lambda tup: tup[1], reverse=True)
    sorted_centers = np.array([tup[0] for tup in sorted_by_intensity])
    unique = np.ones(len(sorted_centers), dtype=np.bool)
    
    with tf.Graph().as_default():
        X_t = tf.placeholder(tf.float32, shape=(n_features, ), name='input')
        dist_t = tf.sqrt(tf.reduce_sum(tf.square(X_t-sorted_centers) , axis=1))
        
        with tf.Session() as sess:
            for i, center in enumerate(sorted_centers):
                if unique[i]:
                    dist = sess.run(dist_t , feed_dict={X_t : center}) 
                    neighbor_idxs = np.where(dist < bandwidth)[0]
                    unique[neighbor_idxs] = 0
                    unique[i] = 1  # leave the current point as unique
            cluster_centers = sorted_centers[unique]
    
    # ASSIGN LABELS: a point belongs to the cluster that it is closest to
    labels = np.zeros(n_samples, dtype=np.int)
    n_clusters = cluster_centers.shape[0]
    with tf.Graph().as_default():
        X_t = tf.placeholder(tf.float32, shape=(batch_size, n_features))
        tiled_X_t = tf.tile(tf.expand_dims(X_t, 1) , [1, n_clusters, 1])
        tiled_clusters = tf.tile(tf.expand_dims(cluster_centers, 0), [batch_size, 1, 1])
        dist_t = tf.sqrt(tf.reduce_sum(tf.square(tiled_X_t-tiled_clusters) , axis=2))
        min_dist_t = tf.reduce_min(dist_t, 1)
        idx_t = tf.argmin(dist_t, 1)

        distances = np.zeros(n_samples, dtype=np.float32)
        idxs = np.zeros(n_samples, dtype=np.int32)
        with tf.Session() as sess:
            for i in range(0, n_samples, batch_size):
                print("\rLabeling: %1.1f%%" %(i/n_samples*100.0),end="")
                start = np.min([i, n_samples - batch_size])
                end = np.min([i + batch_size, n_samples])
                distances[start:end], idxs[start:end] = sess.run([min_dist_t, idx_t] , 
                         feed_dict={X_t : X[start:end]})  
                
    if cluster_all:
        labels = idxs.flatten()
    else:
        labels.fill(-1)
        bool_selector = distances.flatten() <= bandwidth
        labels[bool_selector] = idxs.flatten()[bool_selector]
    return cluster_centers, labels

#def estimate_bandwidth(X, quantile=0.3, n_samples=None, random_state=0,
#                       batch_size = 512):
#    """Estimate the bandwidth to use with the mean-shift algorithm.
#
#    That this function takes time at least quadratic in n_samples. For large
#    datasets, it's wise to set that parameter to a small value.
#
#    Parameters
#    ----------
#    X : array-like, shape=[n_samples, n_features]
#        Input points.
#
#    quantile : float, default 0.3
#        should be between [0, 1]
#        0.5 means that the median of all pairwise distances is used.
#
#    n_samples : int, optional
#        The number of samples to use. If not given, all samples are used.
#
#    random_state : int, RandomState instance or None, optional (default=None)
#        If int, random_state is the seed used by the random number generator;
#        If RandomState instance, random_state is the random number generator;
#        If None, the random number generator is the RandomState instance used
#        by `np.random`.
#
#    n_jobs : int, optional (default = 1)
#        The number of parallel jobs to run for neighbors search.
#        If ``-1``, then the number of jobs is set to the number of CPU cores.
#
#    Returns
#    -------
#    bandwidth : float
#        The bandwidth parameter.
#    """
#    X = check_array(X)
#
#    random_state = check_random_state(random_state)
#    if n_samples is not None:
#        idx = random_state.permutation(X.shape[0])[:n_samples]
#        X = X[idx]
##    nbrs = NearestNeighbors(n_neighbors=int(X.shape[0] * quantile),
##                            n_jobs=n_jobs)
#    
#    bandwidth = 0
#    with tf.Graph().as_default():
#        X_t = tf.placeholder(tf.float32, shape=(n_features, ), name='input')
#        X_placeholder_t = tf.placeholder(tf.float32, shape=(X.shape), name='data')
#        X_var_t = tf.Variable(X_placeholder_t)        
#        dist_t = tf.sqrt(tf.reduce_sum(tf.square(X_t-X_var_t) , axis=1))
#        
#        with tf.Session() as sess:
#            init = tf.global_variables_initializer()
#            sess.run(init, {X_placeholder_t:X})
#            for i in range(0, X.shape[0], batch_size): 
#                points_within, my_mean = sess.run(dist_t , feed_dict={X_t : seed}) 
#    
#    
##    nbrs.fit(X)
##
##    bandwidth = 0.
##    for batch in gen_batches(len(X), 500):
##        d, _ = nbrs.kneighbors(X[batch, :], return_distance=True)
##        bandwidth += np.max(d, axis=1).sum()
#
#    return bandwidth / X.shape[0]