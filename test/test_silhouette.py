# write your silhouette score unit tests here
import numpy as np
import pytest
from cluster.silhouette import Silhouette
from sklearn.metrics import silhouette_score as skl_silhouette_score

def test_silhouette_perfect_clusters():
    X = np.array([
        [0, 0],
        [0, 0.1],
        [10, 10],
        [10.1, 10]
    ])
    y = np.array([0, 0, 1, 1])
    sil = Silhouette()
    scores = sil.score(X, y)
    assert len(scores) == 4
    # expect positive silhouette scores near 1 for both clusters
    assert np.all(scores >= -1) and np.all(scores <= 1)
    assert scores[0] > 0.5 and scores[2] > 0.5

def test_silhouette_singleton_cluster():
    X = np.array([
        [0, 0],
        [1, 1],
        [2, 2]
    ])
    y = np.array([0, 0, 1])
    sil = Silhouette()
    scores = sil.score(X, y)
    # Point in singleton cluster should get score 0
    assert scores[2] == 0

def test_silhouette_one_cluster_only():
    X = np.random.rand(5, 2)
    y = np.zeros(5, dtype=int)
    sil = Silhouette()
    scores = sil.score(X, y)
    # All silhouette scores should be 0 if only one cluster
    assert np.all(scores == 0)

def test_silhouette_invalid_shapes():
    sil = Silhouette()
    X = np.random.rand(5, 2)
    y = np.array([0, 1, 2, 0, 1, 2])
    with pytest.raises(Exception):
        sil.score(X, y[:4])

def test_silhouette_negative_score():
    # Two clusters close together (some negative silhouette)
    X = np.array([
        [0, 0], [0, 0.2], [0.1, -0.1],   # cluster 0
        [0, 0.8], [0.12, 1.0], [0, 1.1]  # cluster 1, very near to cluster 0
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    sil = Silhouette()
    scores = sil.score(X, y)
    # Silhouette can be negative if clusters overlap/are close
    assert np.any(scores < 0)
    assert np.all(scores >= -1) and np.all(scores <= 1)

def test_silhouette_compare_sklearn():
    # Random but clearly separable clusters
    X1 = np.random.normal([0, 0], 0.2, size=(25, 2))
    X2 = np.random.normal([5, 5], 0.2, size=(25, 2))
    X = np.concatenate((X1, X2), axis=0)
    y = np.array([0]*25 + [1]*25)
    sil = Silhouette()
    my_scores = sil.score(X, y)
    # sklearn returns a mean silhouette score
    skl_score = skl_silhouette_score(X, y)
    # Silhouette class returns scores for each sample
    my_mean_score = np.mean(my_scores)
    # Check that the average silhouette score matches scikit-learn within a small tolerance
    assert np.allclose(my_mean_score, skl_score, atol=1e-2)
