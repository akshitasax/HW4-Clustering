import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        self.k = k
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.tol = tol
        if tol <= 0:
            raise ValueError("tol must be a positive float")
        self.max_iter = max_iter
        if max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.
        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        init_idx = np.random.choice(mat.shape[0], self.k, replace=False)
        self.centroids = mat[init_idx] # mat[idx] is a row and hence a point in euclidean space
        # self.centroids is a collection of k points, each an array of [1, n] shape, hence [k,n]
        # while the mat is of [m,n] shape. Thus we can use it in cdist
        # outputs distances of [m,k] shape

        # loop through max_iter times, and for each iteration, update the centroids
        for i in range(self.max_iter):
            self.centroids_old = self.centroids.copy() # copy the centroids to be old centroids
            # calculate the distance between the centroids and the data points
            distances = cdist(mat, self.centroids, 'euclidean') # matrix of [m,k] shape
            # assign the data points to centroid index with the min distance amongst k columns
            labels = np.argmin(distances, axis=1)
            # update the centroids to be the mean of the data points assigned to each centroid
            for j in range(self.k):
                self.centroids[j] = np.mean(mat[labels == j], axis=0)

            # calculate difference between old and new centroids
            diff = self.centroids - self.centroids_old
            # np.linalg.norm gets euclidian norm or length of the difference vector
            if np.linalg.norm(diff) < self.tol:
                break
                

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        # take self.centroids and calculate distances of each of the input points to each of the centroids
        distances = cdist(mat, self.centroids, 'euclidean')
        # assign the data points to centroid index with the min distance amongst k columns
        labels = np.argmin(distances, axis=1)
        self.labels = labels

        self.error = np.mean(np.square(mat - self.centroids))

        return labels

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self.error


    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centroids