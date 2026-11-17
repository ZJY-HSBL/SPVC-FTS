"""Paper-topic experiments; see README for datasets and assumptions."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.special import ndtr,ndtri
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spvc_fts import fit,select_bandwidth,financial_returns,errors,msm
ROOT=Path(__file__).resolve().parents[1]

def read(name):
    with (ROOT/'data'/name).open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

def prep():
    cases={}
    r=read('paper_shenzhen.csv')
    cases['shenzhen']=(np.array([v['date'] for v in r[1:]]),financial_returns([float(v['low']) for v in r],[float(v['high']) for v in r]), {'input_rows':len(r),'series':'Shenzhen Component Index','period':'2023-02-24..2023-12-15'})
    r=read('paper_gold.csv')
    cases['gold']=(np.array([v['Time'] for v in r[1:]]),financial_returns([float(v['Low']) for v in r],[float(v['High']) for v in r]), {'input_rows':len(r),'series':'GOLD D1 broker bars','period':'2023-07-19..2024-09-19'})
    r=read('shanghai_2017.csv'); raw=np.array([float(v['temperature_2m_mean']) for v in r]); x=np.diff(raw); mu=x.mean(); sigma=x.std(); orig=ndtr((x-mu)/sigma)
    # The paper formula is infeasible when the tail mass is <= 0.05. Inflate
    # sigma only enough to put every observed point within the finite domain.
    feasible_sigma=max(sigma, np.max(np.abs(x-mu))/ndtri(.95)*1.05)
    p=ndtr((x-mu)/feasible_sigma)
    l=x-(mu+feasible_sigma*ndtri(p-.05)); u=(mu+feasible_sigma*ndtri(p+.05))-x
    cases['shanghai']=(np.array([v['date'] for v in r[1:]]),np.column_stack((x,l,u)),{'input_rows':len(r),'series':'Shanghai mean temperature first differences','period':'2017-05-01..2017-08-31','original_normal_infeasible_count':int(np.count_nonzero((orig<=.05)|(orig>=.95))),'original_sigma':sigma,'used_sigma':feasible_sigma})
    return cases

def main():
    (ROOT/'figures').mkdir(exist_ok=True);(ROOT/'results').mkdir(exist_ok=True)
    summary={}
    for name,(date,data,meta) in prep().items():
        # Explicit practical grid, not inferred from the paper.
        grid=[.1,.2,.3]; lags=(3,3,3)
        print('CV stage1',name,flush=True)
        cv1=select_bandwidth(data,grid,stage=1,lags=lags)
        print('CV stage2',name,flush=True)
        cv2=select_bandwidth(data,grid,stage=2,lags=lags,h_np=cv1['bandwidth'])
        result=fit(data,lags=lags,h_np=cv1['bandwidth'],h_vc=cv2['bandwidth'])
        actual=result['actual']; dates=date[3:]
        scores={}
        for stage in ('stage1','stage2'):
            pred=result[stage]; score=errors(actual,pred)
            score['nonpositive_spreads']=int(np.count_nonzero(pred[:,1:]<=0))
            score['msm']=msm(actual,pred) if not score['nonpositive_spreads'] else None
            scores[stage]=score
        summary[name]={**meta,'lags':lags,'kernel':'gaussian','stage1_cv':cv1,'stage2_cv':cv2,'scores':scores,'fitted_rows':len(actual)}
        with (ROOT/'results'/f'case_{name}.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['date']+[p+'_'+n for p in ('observed','stage1','stage2') for n in ('center','left','right')]);
            for i,d in enumerate(dates):w.writerow([d,*actual[i],*result['stage1'][i],*result['stage2'][i]])
        x=np.arange(len(actual)); fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,constrained_layout=True)
        for j,ax in enumerate(axes):
            for y,label,c in [(actual,'Observed','#1e293b'),(result['stage1'],'Stage 1','#e69c35'),(result['stage2'],'Stage 2','#6253c7')]:ax.plot(x,y[:,j],label=label,color=c,lw=1.1,alpha=.9)
            ax.grid(alpha=.2);ax.set_ylabel(['Centre','Left spread','Right spread'][j])
        axes[0].legend(ncol=3,frameon=False);axes[0].set_title(f'{name.title()} · in-sample fitted components',loc='left',weight='bold');axes[-1].set_xlabel('Observation index');fig.savefig(ROOT/'figures'/f'case_{name}_fit.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(12,5.6),sharex=True,constrained_layout=True)
        for ax,stage,color in zip(axes,['stage1','stage2'],['#e69c35','#6253c7']):
            e=actual[:,0]-result[stage][:,0];ax.axhline(0,color='#64748b',lw=.8);ax.plot(x,e,color=color,lw=1);ax.set_ylabel(stage+' residual');ax.grid(alpha=.2)
        axes[0].set_title(f'{name.title()} · centre residuals (in-sample)',loc='left',weight='bold');axes[-1].set_xlabel('Observation index');fig.savefig(ROOT/'figures'/f'case_{name}_residual.png',dpi=150);plt.close(fig)
        print(name,json.dumps(scores),flush=True)
    (ROOT/'results'/'paper_cases_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
if __name__=='__main__':main()
