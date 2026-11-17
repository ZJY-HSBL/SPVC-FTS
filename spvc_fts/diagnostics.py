"""Transparent centre-only diagnostics and benchmark, not paper's fuzzy Cook method."""
import math
import numpy as np
from scipy.special import ndtr


def _ar(data, lag):
    x=np.asarray(data,float)
    if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all() or lag<1 or len(x)<=lag+2:
        raise ValueError('expected finite (n,3) and sufficient samples')
    t=np.arange(lag,len(x)); X=np.column_stack([np.ones(len(t))]+[x[t-j,0] for j in range(1,lag+1)])
    if np.linalg.matrix_rank(X)!=X.shape[1] or len(t)<=X.shape[1]:
        raise ValueError('rank-deficient AR design')
    coef=np.linalg.lstsq(X,x[t,0],rcond=None)[0]
    return X,x[t,0],coef


def centre_ar(data,lag=3):
    """OLS AR on centre; spreads zero (crisp prediction), no clipping."""
    X,y,beta=_ar(data,lag)
    return np.column_stack([X@beta,np.zeros((len(y),2))])


def cook_centre(data,lag=3):
    """Standard Cook distance for the centre AR design, illustrative proxy."""
    X,y,b=_ar(data,lag); residual=y-X@b; n,p=X.shape
    leverage=np.sum((X@np.linalg.pinv(X.T@X))*X,axis=1)
    mse=float(residual@residual/(n-p))
    if mse<=0 or np.any(leverage>=1-1e-12): raise ValueError('degenerate Cook distance')
    distance=(residual**2/(p*mse))*leverage/(1-leverage)**2
    center=float(distance.mean()); std=float(distance.std(ddof=1))
    return {'distance':distance,'mean':center,'lower':center-3*std,'upper':center+3*std}


def median_runs(values):
    """Two-sided normal-approximation runs test about sample median; ties removed."""
    x=np.asarray(values,float)
    if x.ndim!=1 or not np.isfinite(x).all(): raise ValueError('expected 1-D finite sequence')
    median=np.median(x); signs=np.sign(x-median); signs=signs[signs!=0]
    n1=int(np.count_nonzero(signs>0));n0=int(np.count_nonzero(signs<0));n=n1+n0
    if n0==0 or n1==0 or n<3: raise ValueError('not enough observations on each side of median')
    runs=int(1+np.count_nonzero(signs[1:]!=signs[:-1]))
    mean=1+2*n1*n0/n
    variance=2*n1*n0*(2*n1*n0-n)/(n*n*(n-1))
    if variance<=0: raise ValueError('degenerate runs variance')
    z=(runs-mean)/math.sqrt(variance)
    return {'n':n,'above':n1,'below':n0,'runs':runs,'z':float(z),'p_value':float(2*ndtr(-abs(z))),'median':float(median)}
