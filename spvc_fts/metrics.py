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


def msm(actual, fitted):
    """Mean fuzzy Jaccard similarity for triangular fuzzy numbers.

    Exact piecewise-linear integration of min and max memberships.
    Both left and right spreads must be strictly positive.
    """
    a,b=np.asarray(actual,float),np.asarray(fitted,float)
    errors(a,b)
    if np.any(a[:,1:]<=0) or np.any(b[:,1:]<=0):
        raise ValueError('MSM requires positive triangular spreads')
    def member(x,r):
        c,l,u=r
        return max(0., 1-(c-x)/l) if x<=c else max(0.,1-(x-c)/u)
    sims=[]
    for r,s in zip(a,b):
        knots=sorted(set([r[0]-r[1],r[0],r[0]+r[2],s[0]-s[1],s[0],s[0]+s[2]]))
        points=list(knots)
        for x,y in zip(knots,knots[1:]):
            d0=member(x,r)-member(x,s); d1=member(y,r)-member(y,s)
            if d0*d1<0: points.append(x+(y-x)*d0/(d0-d1))
        points.sort(); inter=union=0.
        for x,y in zip(points,points[1:]):
            ra,rb=member(x,r),member(y,r); sa,sb=member(x,s),member(y,s)
            inter+=(y-x)*(min(ra,sa)+min(rb,sb))/2
            union+=(y-x)*(max(ra,sa)+max(rb,sb))/2
        sims.append(inter/union)
    return float(np.mean(sims))
