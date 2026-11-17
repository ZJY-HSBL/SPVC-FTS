"""python -m spvc_fts series.csv --h-np .2 --h-vc .2"""
import argparse, csv, json
import numpy as np
from . import errors, fit, msm, select_bandwidth

def main():
    p=argparse.ArgumentParser(description='Two-stage fuzzy time-series in-sample fitting')
    p.add_argument('csv',help='CSV with center,left,right columns')
    p.add_argument('--lags',type=int,nargs=3,default=[3,3,3],metavar=('P','Q','M'))
    p.add_argument('--h-np',type=float,default=.2); p.add_argument('--h-vc',type=float,default=.2)
    p.add_argument('--kernel-np',default='gaussian',choices=['gaussian','triweight','biweight','epanechnikov'])
    p.add_argument('--kernel-vc',default='gaussian',choices=['gaussian','triweight','biweight','epanechnikov'])
    p.add_argument('--cv-grid',type=float,nargs='+',help='explicit candidate bandwidths; fits are computationally expensive')
    p.add_argument('--output',default='fitted.csv'); p.add_argument('--summary',default='summary.json')
    a=p.parse_args()
    with open(a.csv,newline='',encoding='utf-8-sig') as f:
        data=np.array([[float(row[k]) for k in ('center','left','right')] for row in csv.DictReader(f)])
    lags=tuple(a.lags); cv={}
    if a.cv_grid:
        cv['stage1']=select_bandwidth(data,a.cv_grid,stage=1,lags=lags,kernel_np=a.kernel_np)
        a.h_np=cv['stage1']['bandwidth']
        cv['stage2']=select_bandwidth(data,a.cv_grid,stage=2,lags=lags,h_np=a.h_np,kernel_np=a.kernel_np,kernel_vc=a.kernel_vc)
        a.h_vc=cv['stage2']['bandwidth']
    for item in cv.values():
        item['scores']=[score if np.isfinite(score) else None for score in item['scores']]
    result=fit(data,lags,a.h_np,a.h_vc,a.kernel_np,a.kernel_vc)
    summary={'evaluation':'in-sample fit, not future prediction','lags':lags,'h_np':a.h_np,'h_vc':a.h_vc,'cv':cv}
    for name in ('stage1','stage2'):
        prediction=result[name]; summary[name]=errors(result['actual'],prediction)
        summary[name]['nonpositive_spreads']=int(np.count_nonzero(prediction[:,1:]<=0))
        try: summary[name]['msm']=msm(result['actual'],prediction)
        except ValueError: summary[name]['msm']=None
    headers=['time']+[prefix+'_'+field for prefix in ('actual','stage1','stage2') for field in ('center','left','right')]
    with open(a.output,'w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(headers)
        for i in range(len(result['time'])):
            writer.writerow([result['time'][i],*result['actual'][i],*result['stage1'][i],*result['stage2'][i]])
    with open(a.summary,'w',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
