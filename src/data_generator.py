import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

def generate_linearly_separable(n_samples=100, noise=0.1, random_state=42):
    """
    Generates a 2D dataset that is mostly linearly separable.
    """
    X, y = make_blobs(
        n_samples=n_samples, 
        centers=2, 
        n_features=2, 
        cluster_std=1.0 + noise, 
        random_state=random_state
    )
    # Convert labels 0,1 to -1,1 for SVM compatibility
    y = np.where(y == 0, -1, 1)
    return X, y

def generate_circular_dataset(n_samples=100, noise=0.05, factor=0.5, random_state=42):
    """
    Generates a 2D dataset with a concentric circles pattern (perfect for RBF Kernel testing).
    """
    X, y = make_circles(
        n_samples=n_samples, 
        noise=noise, 
        factor=factor, 
        random_state=random_state
    )
    y = np.where(y == 0, -1, 1)
    return X, y

def generate_moons_dataset(n_samples=100, noise=0.1, random_state=42):
    """
    Generates a 2D dataset in the shape of two interleaving half circles (moons).
    """
    X, y = make_moons(
        n_samples=n_samples, 
        noise=noise, 
        random_state=random_state
    )
    y = np.where(y == 0, -1, 1)
    return X, y
