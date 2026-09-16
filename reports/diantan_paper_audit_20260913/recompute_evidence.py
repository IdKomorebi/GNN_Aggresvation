"""Read stored experiment records; do not import applications or train models.

Run: conda run -n Pytorch310_codex python reports/diantan_paper_audit_20260913/recompute_evidence.py
"""
import ast
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DIANTAN = ROOT.parent / 'diantan'
LAB = DIANTAN / 'sampling_lab'


def write_csv(name, rows):
    if rows:
        with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def metrics(rows, estimates=None):
    t = np.array([r['true'] for r in rows])
    e = np.array([r['est'] for r in rows]) if estimates is None else estimates
    unsafe = t > .7
    accept = e <= .7
    q = [r.get('query_ms', 0) for r in rows if r.get('query_ms', 0) > 0]
    return dict(n=len(t), unique_target_plan=len({(r['target'], r['plan']) for r in rows}),
                mae=float(np.abs(e-t).mean()), worst_underestimate=float(np.min(e-t)),
                false_safe_count=int((unsafe & accept).sum()),
                false_safe_joint=float((unsafe & accept).mean()),
                false_reject_joint=float((~unsafe & ~accept).mean()),
                accepted_count=int(accept.sum()),
                unsafe_among_accepted=float((unsafe & accept).sum()/max(1,accept.sum())),
                query_ms_median=float(np.median(q)) if q else None)


def load(arm, domain='pjm'):
    rows = []
    for p in sorted((LAB/'results').glob(f'bench_*_{arm}.json')):
        b = json.loads(p.read_text())
        new = '::' in b['target']
        if (domain == 'new') != new:
            continue
        for r in b['records']:
            rows.append(dict(r, h_size=b['h_size']))
    return rows


def main():
    inventory=[]
    for ds, filename in [('pjm_2025','pjm_rto_hourly_2025_cleaned.csv'),
                         ('caiso_2025','caiso_2025_hourly_cleaned.csv')]:
        pair=[]
        for f in [DIANTAN/'data_rule'/ds/'data.csv', ROOT/'data/Processed'/filename]:
            with f.open() as h:
                reader=csv.DictReader(h); records=list(reader); columns=reader.fieldnames
            pair.append((columns,records))
            inventory.append(dict(dataset=ds,path=str(f),rows=len(records),columns=len(columns),
                                  sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
        a,b=pair
        diff={'only_diantan':sorted(set(a[0])-set(b[0])), 'only_gnn':sorted(set(b[0])-set(a[0])),
              'common_columns_with_text_differences':{k:sum(x[k]!=y[k] for x,y in zip(a[1],b[1]))
                for k in sorted(set(a[0])&set(b[0])) if any(x[k]!=y[k] for x,y in zip(a[1],b[1]))}}
        (OUT/f'{ds}_column_comparison.json').write_text(json.dumps(diff,indent=2,ensure_ascii=False))
    write_csv('data_inventory.csv',inventory)

    summary=[]
    arms=['phi','phi+x','phi+x+x2','phi+x+x2·clip','prod','prod·alphaTe','prod·twin',
          'prod·twinAlpha','prod4','dis·k0','maxall','tri']
    for arm in arms:
        rows=load(arm)
        for scene in ['A','B']:
            rs=[r for r in rows if r['scene']==scene]
            if rs:summary.append(dict(domain='pjm',arm=arm,scene=scene,**metrics(rs)))
    write_csv('stored_benchmark_metrics.csv',summary)

    # Compare identical target/repeat/plan records only.
    pairs=[]
    for base,other in [('prod·alphaTe','dis·k0'),('prod·twinAlpha','dis·k0'),
                       ('phi','phi+x'),('phi+x','phi+x+x2')]:
        key=lambda r:(r['target'],r['repeat'],r['plan'])
        b={key(r):r for r in load(base)};o={key(r):r for r in load(other)}
        for sc in ['A','B']:
            ks=sorted(k for k in b.keys()&o.keys() if b[k]['scene']==sc)
            if not ks:continue
            assert all(abs(b[k]['true']-o[k]['true'])<1e-10 for k in ks)
            for arm,mp in [(base,b),(other,o)]:
                pairs.append(dict(comparison=f'{base} vs {other}',arm=arm,scene=sc,
                                  **metrics([mp[k] for k in ks])))
    write_csv('paired_benchmark_metrics.csv',pairs)

    # Execute only the pure plan-construction function, with synthetic field names.
    # Fractions/kinds depend on h_size and RNG consumption, not field identities.
    tree=ast.parse((LAB/'bench.py').read_text())
    fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_plans')
    scope={'np':np}
    exec(compile(ast.Module(body=[fun],type_ignores=[]),str(LAB/'bench.py'),'exec'),scope)
    cache={}
    def features(rows, cols):
        mat=[]
        for r in rows:
            n=r['h_size']
            if n not in cache:
                plans=scope['build_plans']([str(i) for i in range(n)],
                    [{'key':k} for k in ['raw','noise_100','noise_200','aggregate_24h','suppress']],
                    np.arange(n,dtype=float),np.random.default_rng(31337))
                cache[n]={p:dict(noise=sum(v.startswith('noise') for v in m.values())/n,
                                 kinds=len(set(m.values()))/5) for p,m in plans.items()}
            f=dict(cache[n][r['plan']],est=r['est'],est2=r['est']**2,one=1)
            mat.append([f[c] for c in cols])
        return np.array(mat)
    pjm=[r for r in load('dis·k0') if r['scene']=='B']
    new=[r for r in load('dis·k0','new') if r['scene']=='B']
    cols=['noise','kinds','one']
    x=features(pjm,cols);y=np.array([r['true']-r['est'] for r in pjm])
    beta=np.linalg.solve(x.T@x+1e-3*np.eye(3),x.T@y)
    coefficients=dict(zip(cols,beta.tolist()))
    (OUT/'calibration_coefficients_recomputed.json').write_text(json.dumps(coefficients,indent=2))
    cal=[];trade=[]
    for domain,rows in [('pjm_in_sample',pjm),('new_transfer',new)]:
        raw=np.array([r['est'] for r in rows]);t=np.array([r['true'] for r in rows])
        versions={'maxpair':raw,'calibrated_exact':np.clip(raw+features(rows,cols)@beta,-.5,1),
                  'calibrated_production_rounded':np.clip(raw+features(rows,cols)@np.array([-.0627,.038,.0023]),-.5,1)}
        if domain=='pjm_in_sample':
            cross=raw.copy()
            for target in sorted({r['target'] for r in rows}):
                held=np.array([r['target']==target for r in rows])
                b=np.linalg.solve(x[~held].T@x[~held]+1e-3*np.eye(3),x[~held].T@y[~held])
                cross[held]=raw[held]+x[held]@b
            versions['loto_calibration']=np.clip(cross,-.5,1)
        for arm,e in versions.items():
            cal.append(dict(domain=domain,arm=arm,**metrics(rows,e)))
            curve=[]
            for off in np.arange(-.05,.201,.0025):
                ee=np.clip(e+off,-.5,1)
                curve.append((((ee>.7)&(t<=.7)).mean(),((ee<=.7)&(t>.7)).mean()))
            c=np.array(curve);order=np.argsort(c[:,0])
            for level in [.005,.01,.02,.03,.05,.08]:
                # Reproduce historical interpolation; it is descriptive, not a deployed threshold.
                val=float(np.interp(level,c[order,0],c[order,1]))
                trade.append(dict(domain=domain,arm=arm,false_reject_level=level,false_safe_interpolated=val))
        if domain=='new_transfer':
            ec=versions['calibrated_production_rounded']
            for target in sorted({r['target'] for r in rows}):
                idx=[i for i,r in enumerate(rows) if r['target']==target]
                for arm,e in [('maxpair',raw),('calibrated_production_rounded',ec)]:
                    cal.append(dict(domain=target,arm=arm,**metrics([rows[i] for i in idx],e[idx])))
    write_csv('calibration_metrics.csv',cal)
    write_csv('calibration_tradeoffs_historical_interpolation.csv',trade)
    print(json.dumps({'coefficients':coefficients,'paired_metrics':pairs[:4],
                      'transfer':cal[:8]},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
