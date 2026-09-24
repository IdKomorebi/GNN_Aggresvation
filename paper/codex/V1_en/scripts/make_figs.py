"""Redesigned empirical figures. Read-only use of experiments 111,112,116–119.
Run with no arguments to generate all seven result figures. No training is performed.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/gnn_codex_mpl')
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import style as S
import names as N
from data import R, summaries, max_v3, _est_sets, DSCOL, SEQ
from make_schematics import arrow,node
S.setup()
OUT=Path(__file__).resolve().parents[1]/'figures'
MARK={'RTS-GMLC':'o','NEM':'^','PJM':'s','CAISO':'D'}

def fig_toy():
    T=pd.read_csv(R('DNN_Aggresvation117/outputs/toy_scores.csv'))
    V=pd.read_csv(R('DNN_Aggresvation117/outputs/toy_truth.csv')).set_index('集合').V
    fig=plt.figure(figsize=(S.TEXTW,4.5))
    gs=fig.add_gridspec(2,2,left=.07,right=.97,bottom=.13,top=.9,height_ratios=[1,1.05],wspace=.32,hspace=.52)
    ax=fig.add_subplot(gs[0,0]);ax.set(xlim=(0,100),ylim=(0,42));ax.axis('off');S.panel(ax,'a','Known dependency structure')
    for x,y,l,c in [(8,32,'A',S.BLUE),(8,13,'B',S.BLUE),(36,23,'×',S.BLUE),(65,23,'Y',S.ORANGE),(95,23,'P',S.AQUA)]:node(ax,x,y,l,c,3)
    arrow(ax,(12,32),(32,24));arrow(ax,(12,13),(32,22));arrow(ax,(40,23),(61,23));arrow(ax,(91,23),(69,23),S.AQUA)
    for x,y,l in [(77,38,'$C_1$'),(96,38,'$C_2$'),(96,5,'D')]:
        arrow(ax,(95,23),(x,y),S.AQUA);node(ax,x,y,l,S.AQUA,3)
    node(ax,36,4,'N',S.MUTED,3);ax.text(44,4,'independent',va='center',fontsize=8,color=S.MUTED)
    ax.text(45,36,'$aAB$',ha='center',fontsize=9);ax.text(80,16,'$bP$',ha='center',fontsize=9)
    ax=fig.add_subplot(gs[0,1]);S.panel(ax,'b','Combinations cross the threshold')
    sets=['A','B','A + B','D','A + B + D','A + B + C1'];y=np.arange(6)[::-1]
    vals=V.loc[sets].values
    ax.barh(y,vals,.62,color=[S.ORANGE if '+' not in x else S.BLUE for x in sets])
    for yy,v in zip(y,vals):ax.text(v+.02,yy,f'{v:.2f}',va='center',fontsize=8.5)
    ax.set(yticks=y,yticklabels=[s.replace(' + ',' + ').replace('C1','$C_1$') for s in sets],xlim=(0,1.1),xlabel='Retrained inferability $V_y(S)$')
    ax.axvline(.7,ls='--',color=S.MUTED,lw=.8);ax.text(.71,5.5,'$\\tau=0.7$',fontsize=8.5)
    ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
    ax=fig.add_subplot(gs[1,:]);S.panel(ax,'c','Which field does each score identify?',y=1.04)
    cols=[('pearson','Pearson'),('mi','MI'),('M0','Single field'),('loco','LOCO'),('perm','Permutation'),('sage','SAGE'),('graph','Graph'),('M2','$M^{(2)}$')]
    m=T[[c for c,_ in cols]].values; norm=np.clip(m,0,None)/np.clip(np.clip(m,0,None).max(0),1e-9,None)
    ax.pcolormesh(np.arange(9)-.5,np.arange(len(T)+1)-.5,norm,cmap=SEQ,vmin=0,vmax=1,shading='flat');ax.set_ylim(len(T)-.5,-.5)
    for i in range(len(T)):
        for j in range(8):ax.text(j,i,f'{0 if abs(m[i,j])<.005 else m[i,j]:.2f}',ha='center',va='center',fontsize=8.5,color='white' if norm[i,j]>.6 else S.INK)
    ax.set(xticks=range(8),xticklabels=[l for _,l in cols],yticks=range(len(T)),yticklabels=T.字段)
    ax.tick_params(length=0);ax.add_patch(plt.Rectangle((6.5,-.5),1,6,fill=False,ec=S.BLUE,lw=1.4))
    for sp in ax.spines.values():sp.set_visible(False)
    S.save(fig,OUT/'fig2_mechanism')

def fig_combination():
    sm=summaries();y=np.arange(len(sm))[::-1]
    fig,axs=plt.subplots(1,3,figsize=(S.TEXTW,3.6),gridspec_kw={'left':.19,'right':.985,'top':.85,'bottom':.31,'wspace':.18,'width_ratios':[1,1,1.1]})
    for ax,tau,l in [(axs[0],.5,'a'),(axs[1],.7,'b')]:
        a=sm[f'τ{tau}_单字段即危险']/sm.p;b=sm[f'τ{tau}_单看安全组合危险']/sm.p
        ax.barh(y,a,.65,color=S.ORANGE,label='Unsafe alone');ax.barh(y,b,.65,left=a,color=S.BLUE,label='Safe alone, critical jointly')
        ax.barh(y,1-a-b,.65,left=a+b,color=S.GRID,label='Not critical')
        ax.set(xlim=(0,1),xticks=[0,.5,1],xticklabels=['0','50','100'],xlabel='Fields (%)',yticks=y,yticklabels=sm.name if l=='a' else ['']*len(y))
        ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False);S.panel(ax,l,f'$\\tau={tau}$')
    ax=axs[2];v3=[max_v3(r) for _,r in sm.iterrows()]
    for yy,v1,v in zip(y,sm.最强单字段V,v3):ax.plot([v1,v],[yy,yy],color=S.MUTED,lw=1)
    ax.scatter(sm.最强单字段V,y,s=24,fc='white',ec=S.ORANGE,label='Best single field',zorder=3)
    ax.scatter(v3,y,s=24,c=S.BLUE,marker='D',label='Best set (≤3 fields)',zorder=3)
    ax.set(xlim=(-.03,1.03),xlabel='Inferability $V_y$',yticks=[]);ax.spines['left'].set_visible(False);ax.grid(axis='x',lw=.5);S.panel(ax,'c','Predictive gain')
    for a in axs:a.set_ylim(-.6,len(sm)-.4)
    h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.55,.1),ncol=3,fontsize=8.5,handlelength=1.1,columnspacing=1)
    h,l=ax.get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.55,.02),ncol=2,fontsize=8.5,handletextpad=.3)
    S.save(fig,OUT/'fig4_combination_risk')

def short(s):
    return N.field(s).replace('System energy price','Energy price').replace('Spin. reserve req.','Spin. reserve').replace('Load fcst. area','Load area').replace('Congestion','Cong.').replace('Wind fcst.','Wind').replace('PV fcst.','PV').replace('Avail. cap.','Capacity')

def fig_profile():
    fig=plt.figure(figsize=(S.TEXTW,6.0));gs=fig.add_gridspec(2,1,left=.24,right=.87,top=.94,bottom=.17,hspace=.72,height_ratios=[1,1.1])
    ax=fig.add_subplot(gs[0]);p=pd.read_csv(R('DNN_Aggresvation111/outputs/analysis/field_profile.csv'))
    d=p[p.目标=='线路潮流_C35'].nlargest(14,'M2').sort_values('M2');y=np.arange(len(d))
    for yy,(_,r) in zip(y,d.iterrows()):ax.plot([r.M0,r.M2],[yy,yy],color='#BFCBD3',lw=1.4)
    ax.scatter(d.r2,y,marker='D',s=17,c=S.MUTED,label='Pearson $r^2$')
    ax.scatter(d.M0,y,s=24,fc='white',ec=S.ORANGE,label='$M^{(0)}$: alone',zorder=3)
    ax.scatter(d.M2,y,s=24,c=S.BLUE,label='$M^{(2)}$: worst background',zorder=4)
    ax.set(yticks=y,yticklabels=[short(x) for x in d.字段],xlim=(0,1),xlabel='Field risk for RTS line C35');ax.tick_params(axis='y',length=0)
    ax.grid(axis='x');ax.spines['left'].set_visible(False);S.panel(ax,'a','Escalation from single-field risk')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.25),ncol=3,fontsize=8,handletextpad=.3,columnspacing=.9)
    ax=fig.add_subplot(gs[1]);g=pd.read_csv(R('DNN_Aggresvation112/outputs/analysis/gain_matrix_机组出力_PPCCGT.csv'),index_col=0).iloc[:8,:8]
    m=g.values;im=ax.pcolormesh(np.arange(9)-.5,np.arange(9)-.5,np.clip(m,0,None),cmap=SEQ,vmin=0,vmax=max(.5,np.nanmax(m)),shading='flat');ax.set_ylim(7.5,-.5)
    for i in range(8):
        for j in range(8):
            if i==j:ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,fill=False,ec=S.ORANGE,lw=1.2))
            if m[i,j]>=.1 or i==j:ax.text(j,i,f'{m[i,j]:.2f}',ha='center',va='center',fontsize=8.5,color='white' if m[i,j]>.35 else S.INK)
    labs=[short(x) for x in g.index];ax.set(xticks=range(8),xticklabels=labs,yticks=range(8),yticklabels=labs,xlabel='Background field $j$',ylabel='Released field $i$')
    plt.setp(ax.get_xticklabels(),rotation=35,ha='right');ax.tick_params(length=0)
    for sp in ax.spines.values():sp.set_visible(False)
    S.panel(ax,'b','Partner-dependent gain: NEM PPCCGT')
    cax=fig.add_axes([.9,.18,.018,.29]);cb=fig.colorbar(im,cax=cax);cb.solids.set_rasterized(False);cb.outline.set_visible(False);cb.set_label('Marginal gain / single-field value',fontsize=8.5)
    S.save(fig,OUT/'fig5_profile')

def fig_definition():
    a=R('DNN_Aggresvation117/outputs/analysis');rules=pd.read_csv(a+'/decision_rules.csv');w=pd.read_csv(a+'/withholding_strategies.csv')
    det=pd.read_csv(a+'/detection.csv')
    fig,axs=plt.subplots(1,3,figsize=(S.TEXTW,3.8),gridspec_kw={'left':.08,'right':.98,'bottom':.3,'top':.83,'wspace':.56,'width_ratios':[1,1.08,1.03]})
    ax=axs[0];x=np.arange(2);width=.23
    for k,(rule,dl,lab,col) in enumerate([('单字段规则 V({i})>τ',None,'Single field',S.ORANGE),('扫描标记（估计）',.05,'Scan',S.LIGHTBLUE),('扫描—认证标记',.05,'Scan + certify',S.BLUE)]):
        d=rules[(rules.规则==rule)&(rules.δ.isna() if dl is None else rules.δ==dl)];m=d.groupby('τ').召回.mean().reindex([.5,.7])
        ax.bar(x+(k-1)*width,m,width*.9,color=col,label=lab)
    ax.set(xticks=x,xticklabels=['0.5','0.7'],xlabel='Threshold $\\tau$',ylabel='Critical-field recall',ylim=(0,1.03));S.ygrid(ax);S.panel(ax,'a','Field flagging')
    ax.legend(loc='upper left',bbox_to_anchor=(-.1,-.29),fontsize=8.5)
    ax=axs[1];order=['loco','sage','M0','graph','mi','M2_est'];names=['LOCO','SAGE','Single','Graph','MI','$\\hat M^{(2)}$'];rp=det.groupby(['τ','方法']).R精确率.mean().unstack(0).reindex(order)
    yy=np.arange(len(order))
    for y,v0,v1 in zip(yy,rp[.5],rp[.7]):ax.plot([v0,v1],[y,y],color=S.GRID,lw=2)
    ax.scatter(rp[.5],yy,s=24,fc='white',ec=S.MUTED,label='$\\tau=0.5$');ax.scatter(rp[.7],yy,s=24,c=S.BLUE,marker='D',label='$\\tau=0.7$')
    ax.set(yticks=yy,yticklabels=names,xlim=(.65,1.01),xlabel='R-precision');ax.tick_params(axis='y',length=0);ax.grid(axis='x');ax.spines['left'].set_visible(False);S.panel(ax,'b','Field ranking')
    ax.legend(loc='upper left',bbox_to_anchor=(-.1,-.29),fontsize=8.5)
    ax=axs[2]
    for rule,dl,lab,col in [('单字段定级（扣留单字段越阈者）',None,'Single field',S.ORANGE),('估计MUS最小命中集',.05,'Hitting set',S.AQUA),('自适应M̂2贪心',.05,'Adaptive greedy',S.BLUE)]:
        d=w[(w.策略==rule)&(w.δ.isna() if dl is None else w.δ==dl)]
        for tau,mk in [(.5,'o'),(.7,'s')]:
            e=d[d.τ==tau];ax.scatter(e.扣留数.mean()-e.最优扣留数.mean(),100*e.残余真危险组合.sum()/e.危险组合数.sum(),s=44,marker=mk,color=col,edgecolor='white',lw=.6,label=lab if tau==.5 else None,zorder=3)
    ax.axvline(0,ls=':',color=S.MUTED,lw=.8);ax.set(xlim=(-7,2),ylim=(-5,105),xlabel='Withheld − optimum',ylabel='Unsafe sets left (%)');S.ygrid(ax);S.panel(ax,'c','Withholding')
    ax.legend(loc='upper left',bbox_to_anchor=(-.15,-.26),fontsize=8.5)
    S.save(fig,OUT/'fig6_definition')

def fig_estimator():
    eall=_est_sets();fig,axs=plt.subplots(1,3,figsize=(S.TEXTW,3.15),gridspec_kw={'left':.07,'right':.99,'bottom':.24,'top':.82,'wspace':.4})
    rng=np.random.default_rng(0);seen=set()
    for e in eall:
        ds=e['ds'];col=DSCOL[ds];lab=ds if ds not in seen else None;seen.add(ds)
        idx=rng.choice(len(e['V']),min(450,len(e['V'])),replace=False)
        axs[0].scatter(e['V'][idx].ravel(),e['Ve'][idx].ravel(),s=4,alpha=.25,c=col,marker=MARK[ds],label=lab,edgecolor='none')
        sel=e['bsz']<=2;mt=e['Dt'][:,sel].max(1);me=e['De'][:,sel].max(1)
        axs[1].scatter(mt.ravel(),me.ravel(),s=13,alpha=.7,c=col,marker=MARK[ds],lw=.2,edgecolor='white')
    for ax,l,title,label in [(axs[0],'a','Set inferability','Retrained $V_y(S)$'),(axs[1],'b','Field risk','Retrained $M^{(2)}$')]:
        ax.plot([0,1],[0,1],ls='--',c=S.MUTED,lw=.8);ax.set(xlim=(0,1),ylim=(0,1),xlabel=label,ylabel='Amortized estimate',xticks=[0,.5,1],yticks=[0,.5,1]);S.panel(ax,l,title)
    ks=np.arange(1,11);ax=axs[2]
    for ds in DSCOL:
        rows=[]
        for e in [v for v in eall if v['ds']==ds]:
            sel=np.where(e['bsz']<=2)[0]
            for c in range(len(e['targ'])):
                dt,de=e['Dt'][:,sel,c],e['De'][:,sel,c];mt=dt.max(1);o=np.argsort(-de,1)
                rows.append([np.take_along_axis(dt,o[:,:k],1).max(1).sum()/max(mt.sum(),1e-9) for k in ks])
        ax.plot(ks,np.mean(rows,0),color=DSCOL[ds],marker=MARK[ds],ms=3,label=ds)
    ax.set(ylim=(.8,1.005),xticks=[1,3,5,10],xlabel='Backgrounds retrained, $k$',ylabel='Certified / exact risk');S.ygrid(ax);S.panel(ax,'c','Witness recovery')
    h,l=ax.get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,bbox_to_anchor=(.52,.025),handlelength=1.4)
    S.save(fig,OUT/'fig7_estimator')

def fig_ablation():
    v=pd.read_csv(R('DNN_Aggresvation118/outputs/analysis/variants_by_target.csv'))
    order=[('lin','Linear; no backbone'),('polyS','Quadratic; no backbone'),('head3','Shared output head'),('rand1','Random backbone'),('masknone1','No pre-training masks'),('maskbern1','Bernoulli masks'),('phionly1','Readout on $\\phi$ only'),('nofix1','No numerical safeguards'),('D1','Proposed; one seed'),('E','Proposed; three seeds')]
    fig,axs=plt.subplots(1,2,figsize=(S.TEXTW,3.8),gridspec_kw={'left':.29,'right':.97,'bottom':.19,'top':.88,'wspace':.25})
    y=np.arange(len(order))[::-1]
    for ax,key,l,title,xlim in [(axs[0],'M2误差','a','Field-risk error',(0,.165)),(axs[1],'认证前1','b','Top-1 witness recovery',(.6,1))]:
        m=v.groupby('变体')[key].mean();per=v.groupby(['变体','数据'])[key].mean()
        for yy,(variant,label) in zip(y,order):
            col=S.BLUE if variant in ('D1','E') else '#B9C6CE'
            ax.barh(yy,m[variant],.6,color=col)
            values=per.loc[variant].values;ax.scatter(values,[yy]*len(values),s=11,fc='white',ec=S.INK2,lw=.65,zorder=3)
        ax.set(yticks=y,yticklabels=[label for _,label in order] if l=='a' else ['']*len(y),xlim=xlim,xlabel='MAE of $M^{(2)}$' if l=='a' else 'Certified / exact risk')
        ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False);ax.grid(axis='x');ax.set_axisbelow(True);S.panel(ax,l,title)
    fig.text(.29,.035,'Bars: mean across ten targets.  ○: mean within each data group.',fontsize=8.5)
    S.save(fig,OUT/'fig8_ablation')

def fig_robust():
    a=R('DNN_Aggresvation119/outputs/analysis');sm=summaries();nem=pd.read_csv(a+'/nem_k3.csv');nem=nem[nem.口径.str.startswith('规模 ≤4')].set_index('目标')
    rows=[]
    for _,r in sm.iterrows():
        m23=float(nem.loc['机组出力_'+r['目标'].split('_',1)[1],'M2/M3']) if r.ds=='NEM' else r['K饱和_M2除M3']
        rows.append((r['name'],r['K饱和_M1除M2'],m23))
    fig=plt.figure(figsize=(S.TEXTW,5.5));gs=fig.add_gridspec(2,2,left=.19,right=.98,bottom=.18,top=.92,hspace=.85,wspace=.43,height_ratios=[1.3,1])
    ax=fig.add_subplot(gs[0,0]);y=np.arange(len(rows))[::-1]
    ax.scatter([r[1] for r in rows],y,s=24,fc='white',ec=S.ORANGE,label='$M^{(1)}/M^{(2)}$');ax.scatter([r[2] for r in rows],y,s=24,c=S.BLUE,marker='D',label='$M^{(2)}/M^{(3)}$')
    ax.set(yticks=y,yticklabels=[r[0] for r in rows],xlim=(.65,1.02),xlabel='Ratio of mean marginal risk');ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False);ax.grid(axis='x');S.panel(ax,'a','Marginal-risk coverage')
    ax.legend(loc='upper left',bbox_to_anchor=(-.15,-.25),ncol=2,fontsize=8,handletextpad=.2,columnspacing=.7)
    ax=fig.add_subplot(gs[0,1]);k=pd.read_csv(a+'/k3_critical.csv');k=k[k.τ==.7].reset_index(drop=True)
    # Match target order explicitly, rather than relying on source row order.
    k=k.assign(short=k.目标.map(N.target)).set_index('short').loc[[r[0] for r in rows]]
    for yy,(_,r) in zip(y,k.iterrows()):ax.plot([r.K2关键/r.字段数,r.K3关键/r.字段数],[yy,yy],color='#B9C6CE')
    ax.scatter(k.K2关键/k.字段数,y,s=24,c=S.BLUE,label='$K=2$');ax.scatter(k.K3关键/k.字段数,y,s=26,fc='white',ec=S.VIOLET,marker='s',label='$K=3$')
    ax.set(yticks=y,yticklabels=['']*len(y),xlim=(-.03,1.03),xlabel='Critical-field share at $\\tau=0.7$');ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False);ax.grid(axis='x');S.panel(ax,'b','Threshold-crossing coverage')
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.25),ncol=2,fontsize=8)
    ax=fig.add_subplot(gs[1,0]);c=pd.read_csv(a+'/robust_consistency.csv');mets=[('M2排序Spearman','Risk rank\n(Spearman)'),('关键集合Jaccard_τ05','Critical set\n$\\tau=0.5$'),('关键集合Jaccard_τ07','Critical set\n$\\tau=0.7$')]
    for j,(key,label) in enumerate(mets):
        for kind,mk,col,off in [('种子','o',S.BLUE,-.12),('划分','s',S.ORANGE,.12)]:
            v=c[c.版本.str.startswith(kind)][key];ax.scatter(v,j+off+np.linspace(-.04,.04,len(v)),s=20,marker=mk,c=col,label=('Seed' if kind=='种子' else 'Split') if j==0 else None)
    ax.set(yticks=range(3),yticklabels=[l for _,l in mets],xlim=(.9,1.005),ylim=(2.5,-.5),xlabel='Agreement with main run');ax.grid(axis='x');ax.tick_params(axis='y',length=0);S.panel(ax,'c','Seed and split stability')
    ax.legend(loc='upper left',bbox_to_anchor=(-.05,-.3),ncol=2,fontsize=8.5)
    ax=fig.add_subplot(gs[1,1]);t=pd.read_csv(a+'/tau_sweep.csv')
    m=t.assign(c=t.组合危险字段/t.字段数,s=t.单字段即危险/t.字段数).groupby('τ')[['c','s','仍暴露比例']].mean()
    ax.plot(m.index,m.c,c=S.BLUE,marker='o',ms=3,label='Safe alone, critical jointly');ax.plot(m.index,m.s,c=S.ORANGE,ls='--',marker='s',ms=3,label='Unsafe alone');ax.plot(m.index,m.仍暴露比例,c=S.MUTED,ls=':',marker='^',ms=3,label='Unsafe sets left by single-field rule')
    ax.set(xlabel='Threshold $\\tau$',ylabel='Share',ylim=(0,1.02));S.ygrid(ax);S.panel(ax,'d','Threshold sensitivity')
    ax.legend(loc='upper left',bbox_to_anchor=(-.13,-.3),fontsize=8)
    S.save(fig,OUT/'fig9_robustness')

if __name__=='__main__':
    for name in sys.argv[1:] or ['toy','combination','profile','definition','estimator','ablation','robust']:
        globals()['fig_'+name]();print('Generated',name,flush=True)
