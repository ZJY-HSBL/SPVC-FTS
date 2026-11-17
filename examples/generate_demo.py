"""Create an artificial fuzzy series solely for smoke testing."""
import csv
import numpy as np
rng=np.random.default_rng(7)
n=60
x=np.zeros((n,3)); x[0]=[.2,.5,.7]
for i in range(1,n):
    c,l,u=x[i-1]
    x[i]=[.65*c+.08*np.sin(i/6)+rng.normal(0,.015),
          .2+.55*l+.03*c+rng.normal(0,.005),
          .22+.6*u+.02*c+rng.normal(0,.005)]
with open('examples/synthetic.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['center','left','right']); w.writerows(x)
print('Wrote examples/synthetic.csv (artificial data)')
