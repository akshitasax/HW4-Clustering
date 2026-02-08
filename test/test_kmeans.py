# Write your k-means unit tests here
import numoy as np
import pytest
from cluster.kmeans import KMeans
from cluster.utils import make_clusters

def test_kmeans():
    # Test basic KMeans clustering on synthetic data
    mat, true_labels = make_clusters(n=300, m=2, k=3, seed=123)
    kmeans = KMeans(k=3)
    kmeans.fit(mat)
    pred_labels = kmeans.predict(mat)
    # There should be exactly 3 clusters found (label indices may not match but count should)
    assert len(np.unique(pred_labels)) == 3
    # Centroids shape should be (3, 2)
    centroids = kmeans.get_centroids()
    assert centroids.shape == (3, 2)
    # Error should be non-negative and finite
    error = kmeans.get_error()
    assert error >= 0
    assert np.isfinite(error)

    # Test that fit does not raise on straightforward data
    try:
        kmeans.fit(mat)
    except Exception as e:
        pytest.fail(f"KMeans.fit raised an exception: {e}")

    # Test ValueError for invalid k
    with pytest.raises(ValueError):
        KMeans(k=0)
    with pytest.raises(ValueError):
        KMeans(k=-2)

    # Test ValueError for tol
    with pytest.raises(ValueError):
        KMeans(k=3, tol=0)
    with pytest.raises(ValueError):
        KMeans(k=3, tol=-1)

    # Test ValueError for max_iter
    with pytest.raises(ValueError):
        KMeans(k=3, max_iter=0)
    with pytest.raises(ValueError):
        KMeans(k=3, max_iter=-10)

    # Test that prediction shape matches n_samples
    preds = kmeans.predict(mat)
    assert preds.shape == (mat.shape[0],)

    # Test that centroids remain consistent after multiple fit-predict cycles
    kmeans2 = KMeans(k=3)
    kmeans2.fit(mat)
    centroids1 = kmeans2.get_centroids().copy()
    kmeans2.fit(mat)
    centroids2 = kmeans2.get_centroids()
    assert centroids1.shape == centroids2.shape

    # Edge case: all points in a cluster
    mat_same = np.ones((10, 2))
    kmeans_same = KMeans(k=1)
    kmeans_same.fit(mat_same)
    centroids_same = kmeans_same.get_centroids()
    assert np.allclose(centroids_same, mat_same[0])

    # Test predict on bad shape (different number of features)
    mat_bad = np.ones((10, 3))
    with pytest.raises(ValueError):
        kmeans.predict(mat_bad)
