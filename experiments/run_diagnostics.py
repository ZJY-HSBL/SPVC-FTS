"""Supplementary diagnostics and baselines for paper-topic cases."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.run_paper_cases import prep
from spvc_fts import fit,errors,msm
from spvc_fts.diagnostics import centre_ar,cook_centre,median_runs
ROOT=Path(__file__).resolve().parents[1]
KINDS=['gaussian','epanechnikov','biweight','triweight']

def main():
    results={}
    for name,(dates,data,meta) in prep().items():
        print('Running',name,flush=True)
        ar=centre_ar(data,3); actual=data[3:]; compare={'centre_ar':{**errors(actual,ar),'msm':0.0}}
        diagnosis=cook_centre(data,3)
        runs=median_runs(data[:,0]);times=np.arange(len(actual))
        with (ROOT/'results'/f'case_{name}.csv').open(newline='') as f: rows=list(csv.DictReader(f))
        for stage in ['stage1','stage2']:
            pred=np.array([[float(row[stage+'_'+c]) for c in ('center','left','right')] for row in rows])
            compare[stage]={**errors(actual,pred),'msm':msm(actual,pred) if np.all(pred[:,1:]>0) else None}
        cv=[]; gaussian=None
        for kind in KINDS:
            print(' ',kind,flush=True)
            r=fit(data,lags=(3,3,3),h_np=.3,h_vc=.3,kernel_np=kind,kernel_vc=kind)
            if kind=='gaussian':gaussian=r
            pred=r['stage2'];cv.append({'kernel':kind,**errors(actual,pred),'invalid_spreads':int(np.sum(pred[:,1:]<=0)),'msm':msm(actual,pred) if np.all(pred[:,1:]>0) else None})
        flagged=np.flatnonzero((diagnosis['distance']>diagnosis['upper'])|(diagnosis['distance']<diagnosis['lower']))
        results[name]={'baseline':'OLS centre AR(3), crisp forecast; full-sample fit','comparison':compare,'runs':runs,'cook_proxy':{'mean':diagnosis['mean'],'lower':diagnosis['lower'],'upper':diagnosis['upper'],'count':len(flagged),'dates':[str(dates[i+3]) for i in flagged]},'fixed_bandwidth_kernel_sensitivity':cv}
        fig,ax=plt.subplots(figsize=(12,3.7),constrained_layout=True)
        ax.plot(times,diagnosis['distance'],color='#6253c7',lw=1.1);ax.axhline(diagnosis['mean'],color='#4099a0',ls='--',label='Mean')
        ax.axhline(diagnosis['upper'],color='#df704e',ls='--',label='Mean + 3 SD');ax.scatter(flagged,diagnosis['distance'][flagged],s=30,color='#df704e',zorder=3,label='Flagged')
        ax.set_title(f'{name.title()} · centre AR Cook distance (proxy)',loc='left',weight='bold');ax.set_xlabel('Observation index');ax.set_ylabel("Cook's D");ax.grid(alpha=.2);ax.legend(frameon=False);fig.savefig(ROOT/'figures'/f'case_{name}_cook.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(3,1,figsize=(12,7),sharex=True,constrained_layout=True)
        for j,ax in enumerate(axes):
            ax.plot(times,gaussian['varying_coefficients'][:,j],color='#6253c7',label='Stage 2')
            ax.axhline(gaussian['constant_coefficients'][j],ls='--',color='#df9b30',label='Stage 1')
            ax.set_ylabel(f'Lag {j+1}');ax.grid(alpha=.2)
        axes[0].set_title(f'{name.title()} · centre coefficients (Gaussian, h=0.3)',loc='left',weight='bold');axes[0].legend(frameon=False,ncol=2);axes[-1].set_xlabel('Observation index');fig.savefig(ROOT/'figures'/f'case_{name}_coefficients.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,3.6),constrained_layout=True)
        values=[k['mse'] for k in cv];ax.bar([k['kernel'] for k in cv],values,color=['#6253c7','#8b7de3','#aaa0ec','#d0c8f7']);ax.set_ylabel('MSE');ax.set_title(f'{name.title()} · fixed-bandwidth kernel sensitivity',loc='left',weight='bold');ax.grid(axis='y',alpha=.2)
        fig.savefig(ROOT/'figures'/f'case_{name}_kernels.png',dpi=150);plt.close(fig)
        print(name,json.dumps(results[name]),flush=True)
    (ROOT/'results'/'diagnostics_summary.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
if __name__=='__main__':main()
