import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
        self.silhouette_scores = None # initialize to None

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        s_list = []
        for i in range(X.shape[0]):

            s = np.zeros(X.shape[0], dtype=float)
            cluster_labels = np.unique(y)

            cluster_idx = y[i]
            same_idx = np.where(y == cluster_idx)[0]

            if same_idx.size <= 1:
                a = 0.0
            else:
                same_others = same_idx[same_idx != i]

                # a is the distance of each point from other members of their cluster, averaged
                a = cdist(X[i:i+1], X[same_others], "euclidean").mean()

            # b: calculate mean distance to all other clusters, and then b is the smallest distance 
            mean_distances = []
            b = np.inf
            for l in cluster_labels:
                if l == cluster_idx: # skip own cluster
                    continue
                other_idx = np.where(y == l)[0]
                if other_idx.size == 0:
                    continue
                mean_dist = cdist(X[i:i+1], X[other_idx], "euclidean").mean()
                b = min(b, mean_dist)

            # If no other clusters exist, define silhouette as 0
            if not np.isfinite(b) or (a == 0.0 and b == 0.0):
                s[i] = 0.0
            else:
                s[i] = (b - a) / max(a, b)

        return s
            
        
