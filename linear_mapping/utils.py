import numpy as np
import numpy.typing as npt
import os
import requests

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

def download_kay():
  """Download dataset from Kay and Gallant"""

  fnames = ["kay_labels.npy", "kay_labels_val.npy", "kay_images.npz"]
  urls = ["https://osf.io/r638s/download",
      "https://osf.io/yqb3e/download",
      "https://osf.io/ymnjv/download"]

  for fname, url in zip(fnames, urls):
    if not os.path.isfile(fname):
      try:
        r = requests.get(url)
      except requests.ConnectionError:
        print("!!! Failed to download data !!!")
      else:
        if r.status_code != requests.codes.ok:
          print("!!! Failed to download data !!!")
        else:
          print(f"Downloading {fname}...")
          with open(fname, "wb") as fid:
            fid.write(r.content)
          print(f"Download {fname} completed!")

def kay_exists() -> bool:
  """Check that Kay and Gallant dataset exists"""

  fnames = ["kay_labels.npy", "kay_labels_val.npy", "kay_images.npz"]

  return all(map(os.path.exists, fnames))

def load_kay() -> dict | np.ndarray | np.ndarray:
  """Load dataset from Kay and Gallant into workspace"""

  with np.load( "kay_images.npz") as dobj:
    dat = dict(**dobj)
    labels = np.load('kay_labels.npy')
    val_labels = np.load('kay_labels_val.npy')

  return dat, labels, val_labels