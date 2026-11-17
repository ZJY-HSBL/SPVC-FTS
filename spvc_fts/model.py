"""Two-stage in-sample fitting; see README for assumptions and CV limitations."""
import numpy as np
from scipy.optimize import least_squares


def kernel(x, name='gaussian'):
    x = np.asarray(x, float)
    if name == 'gaussian': return np.exp(-0.5*x*x)
    powers = {'epanechnikov': 1, 'biweight': 2, 'triweight': 3}
    if name not in powers: raise ValueError('unknown kernel')
    return np.maximum(1-x*x, 0)**powers[name]


def _design(data, lags):
    a = np.asarray(data, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all() or np.any(a[:,1:] < 0):
        raise ValueError('expected finite (n,3) centre and non-negative spreads')
    if len(lags) != 3 or any(int(v) != v or v < 1 for v in lags):
        raise ValueError('lags must contain three positive integers')
    p,q,m = lags; k=max(lags); t=np.arange(k,len(a))
    if len(t) <= max(p,q+2,m+2): raise ValueError('too few observations')
    X = np.column_stack([a[t-i,0] for i in range(1,p+1)])
    L = np.column_stack([np.ones(len(t))]+[a[t-i,1] for i in range(1,q+1)])
    U = np.column_stack([np.ones(len(t))]+[a[t-i,2] for i in range(1,m+1)])
    return a[t],X,L,U,(t+1)/len(a)


def _smooth(train_t, target_t, bandwidth, kind):
    if not np.isfinite(bandwidth) or bandwidth <= 0: raise ValueError('bandwidth must be positive and finite')
    w=kernel((target_t[:,None]-train_t[None,:])/bandwidth,kind)
    s=w.sum(axis=1,keepdims=True)
    if np.any(s < 1e-14): raise ValueError('kernel has no support at a target')
    return w/s


def _split(v,p,q,m):
    return v[:p],v[p:p+q+2],v[p+q+2:p+q+m+4]


def _predict(theta,X,L,U,trend,p,q,m):
    alpha,beta,phi=_split(theta,p,q,m)
    c=X@alpha+trend
    return np.column_stack([c, L@beta[:-1]+beta[-1]*c, U@phi[:-1]+phi[-1]*c])


def _fit_stage1(y,X,L,U,t,h,kind,shape, max_iter=30):
    p,q,m=shape
    S=_smooth(t,t,h,kind)
    # Equation (3): f(t)=S(y_c-X alpha).
    def pred(theta):
        a=theta[:p]
        return _predict(theta,X,L,U,S@(y[:,0]-X@a),p,q,m)
    theta=np.zeros(p+q+m+4); weights=np.ones(len(y))
    for _ in range(max_iter):
        def residual(v): return ((pred(v)-y)*np.sqrt(weights[:,None]*np.array([2/3,1/3,1/3]))).ravel()
        fit=least_squares(residual,theta,max_nfev=500)
        if not fit.success: raise RuntimeError('stage 1 optimizer failed: '+fit.message)
        dist=np.sum((pred(fit.x)-y)**2*np.array([2/3,1/3,1/3]),axis=1)
        new=1/(1+dist)
        if np.max(np.abs(new-weights)) < 1e-8:
            theta=fit.x; break
        theta,weights=fit.x,new
    return theta


def _trend(y,X,t,target,theta,h,kind):
    return _smooth(t,target,h,kind)@(y[:,0]-X@theta[:X.shape[1]])


def _local(y,X,L,U,t,target,trend,h,kind,initial,shape):
    if not np.isfinite(h) or h <= 0: raise ValueError('bandwidth must be positive and finite')
    weights=kernel((t-target)/h,kind)
    if np.count_nonzero(weights>1e-10) < max(shape[0],shape[1]+2,shape[2]+2):
        raise ValueError('insufficient kernel support for local fit')
    def residual(v):
        return ((_predict(v,X,L,U,trend,*shape)-y)*np.sqrt(weights[:,None]*np.array([2/3,1/3,1/3]))).ravel()
    fit=least_squares(residual,initial,max_nfev=500)
    if not fit.success: raise RuntimeError('stage 2 optimizer failed: '+fit.message)
    return fit.x


def fit(data, lags=(3,3,3), h_np=.2, h_vc=.2, kernel_np='gaussian', kernel_vc='gaussian'):
    """Fit both stages to all observations; output is in-sample, not forecasts."""
    y,X,L,U,t=_design(data,lags)
    theta=_fit_stage1(y,X,L,U,t,h_np,kernel_np,lags)
    f=_trend(y,X,t,t,theta,h_np,kernel_np)
    stage1=_predict(theta,X,L,U,f,*lags)
    coeff=np.array([_local(y,X,L,U,t,ti,f,h_vc,kernel_vc,theta,lags) for ti in t])
    stage2=np.vstack([_predict(v,X[i:i+1],L[i:i+1],U[i:i+1],f[i:i+1],*lags)[0] for i,v in enumerate(coeff)])
    return {'actual':y,'time':t,'trend':f,'stage1':stage1,'stage2':stage2,'constant_coefficients':theta,'varying_coefficients':coeff}


def select_bandwidth(data, candidates, stage=1, lags=(3,3,3), h_np=.2, kernel_np='gaussian', kernel_vc='gaussian'):
    """Leave-one-response-out CV. For stage 2 each fold refits stage 1.

    Lagged predictors use the original observations; this is not rolling-origin CV.
    """
    if stage not in (1,2): raise ValueError('stage must be 1 or 2')
    candidates=list(candidates)
    if not candidates or any(not np.isfinite(h) or h<=0 for h in candidates): raise ValueError('positive finite candidates required')
    y,X,L,U,t=_design(data,lags); scores=[]
    for h in candidates:
        loss=[]
        for i in range(len(y)):
            mask=np.arange(len(y)) != i
            yt,xt,lt,ut,tt=y[mask],X[mask],L[mask],U[mask],t[mask]
            hn=h if stage==1 else h_np
            try:
                theta=_fit_stage1(yt,xt,lt,ut,tt,hn,kernel_np,lags)
                fi=_trend(yt,xt,tt,t[i:i+1],theta,hn,kernel_np)
                if stage==2:
                    ft=_trend(yt,xt,tt,tt,theta,hn,kernel_np)
                    theta=_local(yt,xt,lt,ut,tt,t[i],ft,h,kernel_vc,theta,lags)
                pred=_predict(theta,X[i:i+1],L[i:i+1],U[i:i+1],fi,*lags)[0]
                loss.append(np.sum((pred-y[i])**2*np.array([2/3,1/3,1/3])))
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                loss=[np.inf]; break
        scores.append(float(np.mean(loss)))
    if not np.isfinite(scores).any(): raise ValueError('no candidate produced valid fits')
    return {'bandwidth':float(candidates[int(np.argmin(scores))]),'candidates':candidates,'scores':scores,'stage':stage}
