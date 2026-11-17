"""Error criteria, section 2.3; sums over three components per observation."""
import numpy as np

def errors(actual, fitted):
    a, b = np.asarray(actual, float), np.asarray(fitted, float)
    if a.ndim != 2 or a.shape[1] != 3 or a.shape != b.shape or not len(a):
        raise ValueError('expected matching nonempty (n, 3) arrays')
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('inputs must be finite')
    d = a-b
    return {'mse': float(np.mean(np.sum(d*d, axis=1))),
            'mae': float(np.mean(np.sum(np.abs(d), axis=1)))}
