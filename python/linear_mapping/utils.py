import numpy as np
import numpy.typing as npt

def reshape(data:npt.ArrayLike) -> npt.ArrayLike:
    """Flatten N x M x P matrix to (N * M) x P"""
    return data.reshape(-1, data.shape[-1])

def fake_data(n_samples:int, n_features:int, s:np.float32) -> npt.ArrayLike | npt.ArrayLike | npt.ArrayLike:
    """Generate fake 2 dimensional data
    
    Parameters
    ----------
    
    n_samples:int
        Number of row vectors
        
    n_features:int
        Number of column vectors

    s: np.float32
        Variance
        
    Return
    ------
    A: npt.ArrayLike
    
    b: npt.ArrayLike

    W: npt.ArrayLike
    """

    mean = np.zeros(n_features)
    cov = np.eye(n_features)

    A = np.random.multivariate_normal(mean, cov, n_samples)

    W = np.random.rand(n_features)

    s = s * np.random.randn(n_samples)
    b = A @ W + s

    return A, b, W