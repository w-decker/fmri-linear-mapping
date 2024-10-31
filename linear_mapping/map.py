import numpy as np
import numpy.typing as npt
from .utils import reshape

def regsvd(A:npt.ArrayLike, b:npt.ArrayLike) -> npt.ArrayLike:
    """Compute linear mapping between Matrix and and b using economical SVD
    
    Parameters
    ----------
    A: npt.ArrayLike
        Independent variable
        
    b: npt.ArrayLike
        Dependent variable
        
    Return
    ------
    W: npt.ArrayLike
        Model weights"""
    
    if A.ndim > 2:
        A = reshape(A)
    
    # if b.ndim > 1 and b.shape[1] != 1:
    #     b = b.reshape(-1)

    W = np.linalg.pinv(A) @ b
    
    return W