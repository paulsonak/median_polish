from typing import Dict
import numpy as np

def median_polish(data: np.ndarray, n_iter: int = 10, tol: float = 0.01) -> Dict[str, np.ndarray]:
    """Performs median polish on a 2-D array based on Brideau C and Pipeline Pilot
    Args:
        data: input 2-D array
    Returns:
        a dict, with:
            ave: μ
            col: column effect
            row: row effect
            r: cell residue
    """
    assert data.ndim == 2, "Input must be 2D array"
    ndim = 2
    data = data.copy()
    grand_effect = np.mean(data) # changed to mean to match PLP
    data -= grand_effect
    median_margins = [0] * ndim
    margins = [np.zeros(shape=data.shape[idx]) for idx in range(2)]
    dim_mask = np.ones(ndim, dtype=np.int)
    stop=False
    
    for _ in range(n_iter):                                     # for num of iterations,
        for dim_id in range(ndim):                                # for each dimension,
            rest_dim = 1 - dim_id                                   # define dimension- row or column
            temp_median = np.median(data, rest_dim)                 # get medians of rows or columns aka effect
            margins[dim_id] += temp_median                          # add medians to row/column effects
            median_margins[rest_dim] = np.median(margins[rest_dim]) # get median of row/column effects
            # if median_margins[rest_dim]<tol:
            #     stop=True
            #     break
            margins[rest_dim] -= median_margins[rest_dim]           # subtract median from the effects
            dim_mask[dim_id] = -1                                   # create mask for dimension
            data -= temp_median.reshape(dim_mask)                   # subtract medians along only the row/col dimension by reshaping data
            dim_mask[dim_id] = 1                                    # reset mask
        grand_effect += sum(median_margins)                       # add the median of row and column effects to the grand effect
        if stop:
            break
    return {'ave': grand_effect, 'row': margins[1], 'column': margins[0], 'r': data}


def med_abs_dev(data: np.ndarray) -> float:
    """Median absolute deviation.
    MAD = median(|X_i - median(X)|)
    """
    return np.median(np.abs(data - np.median(data)))
