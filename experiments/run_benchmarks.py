"""Generate reproducible supplemental in-sample benchmarks, not paper replication."""
from pathlib import Path
import csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spvc_fts import fit, errors, msm

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'; FIG=ROOT/'figures'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi':130,'savefig.dpi':150,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def load(name, divisor):
    with (ROOT/'data'/f'{name}.csv').open(newline='') as f:
        rows=[r for r in list(csv.reader(f))[1:] if r]
    year=np.array([int(float(r[0])) for r in rows]); value=np.array([float(r[1]) for r in rows])/divisor
    # Trailing variation is an explicitly chosen illustrative fuzzification.
    # At t, spread uses up to five past adjacent changes, including t's value.
    spread=np.array([.04+np.mean(np.abs(np.diff(value[max(0,i-5):i+1]))) if i else .04 for i in range(len(value))])
    return year,np.column_stack([value,spread,spread])


def run(name, divisor):
    year,data=load(name,divisor)
    result=fit(data,lags=(1,1,1),h_np=.25,h_vc=.25)
    yy=year[1:]; actual=result['actual']; first=result['stage1']; second=result['stage2']
    scores={}
    for label,pred in [('stage1',first),('stage2',second)]:
        scores[label]=errors(actual,pred)
        scores[label]['invalid_spreads']=int(np.count_nonzero(pred[:,1:]<=0))
        scores[label]['msm']=msm(actual,pred) if np.all(pred[:,1:]>0) else None
    with (OUT/f'{name}_fitted.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['year']+[p+'_'+n for p in ('observed','stage1','stage2') for n in ('center','left','right')])
        for yr,a,b,c in zip(yy,actual,first,second): writer.writerow([yr,*a,*b,*c])
    fig,axes=plt.subplots(3,1,figsize=(12,8.2),sharex=True,constrained_layout=True)
    for ax,col,title in zip(axes,range(3),['Centre','Left spread','Right spread']):
        ax.plot(yy,actual[:,col],color='#1e293b',lw=1.3,label='Observed')
        ax.plot(yy,first[:,col],color='#df9b30',lw=1.2,alpha=.9,label='Stage 1')
        ax.plot(yy,second[:,col],color='#6253c7',lw=1.3,alpha=.9,label='Stage 2')
        ax.set_ylabel(title); ax.grid(alpha=.2)
    axes[0].set_title(f'{name.title()} · in-sample fit',loc='left',weight='bold',fontsize=14)
    axes[0].legend(ncol=3,frameon=False,loc='upper right'); axes[-1].set_xlabel('Year')
    fig.savefig(FIG/f'{name}_fit.png',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,6.3),sharex=True,constrained_layout=True)
    for ax,pred,label,color in zip(axes,[first,second],['Stage 1','Stage 2'],['#df9b30','#6253c7']):
        err=actual[:,0]-pred[:,0]
        ax.axhline(0,color='#475569',lw=.8); ax.plot(yy,err,color=color,lw=1.1)
        ax.fill_between(yy,0,err,color=color,alpha=.15);ax.set_ylabel('Centre residual');ax.set_title(label,loc='left');ax.grid(alpha=.2)
    axes[0].set_title(f'{name.title()} · centre residuals, in-sample',loc='left',weight='bold',fontsize=13)
    axes[-1].set_xlabel('Year');fig.savefig(FIG/f'{name}_residuals.png',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,3.8),constrained_layout=True)
    coeff=result['varying_coefficients']; ax.plot(yy,coeff[:,0],color='#6253c7',label='Varying centre lag coefficient')
    ax.axhline(result['constant_coefficients'][0],color='#df9b30',ls='--',label='Stage 1 constant coefficient')
    ax.set_title(f'{name.title()} · estimated centre coefficient',loc='left',weight='bold',fontsize=13)
    ax.set_xlabel('Year');ax.set_ylabel('Coefficient');ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.savefig(FIG/f'{name}_coefficient.png',bbox_inches='tight');plt.close(fig)
    return {'data_points':len(data),'fitted_points':len(actual),'years':[int(year[0]),int(year[-1])], 'scale_divisor':divisor,'lags':[1,1,1],'h_np':.25,'h_vc':.25,'kernel':'gaussian','scores':scores}


def main():
    summary={name:run(name,div) for name,div in [('nile',1000),('sunspots',100)]}
    (OUT/'benchmark_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
