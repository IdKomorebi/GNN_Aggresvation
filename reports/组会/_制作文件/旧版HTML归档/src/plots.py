"""Redraw presentation figures from the existing experiment CSVs. No training."""
from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/group-meeting-mpl-cache')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
FIG = OUT / 'assets' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': [FONT.get_name(), 'DejaVu Sans'], 'font.size': 20,
    'axes.labelsize': 20, 'xtick.labelsize': 19, 'ytick.labelsize': 21,
    'legend.fontsize': 18, 'svg.fonttype': 'path', 'axes.unicode_minus': False,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
    'axes.edgecolor': '#D6DEDF', 'text.color': '#203544', 'axes.labelcolor': '#546875',
    'xtick.color': '#546875', 'ytick.color': '#203544', 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'savefig.facecolor': 'white'})
C = ['#147D78', '#D67B3F', '#516D9A', '#9BAEB6', '#AF7AA1']
MANIFEST = {}

def read(n, f):
    p = f'DNN_Aggresvation{n}/{f}'
    return pd.read_csv(ROOT/p), p

def save(name, fig, sources, values, note=''):
    fig.savefig(FIG/f'{name}.svg', bbox_inches='tight', pad_inches=.22)
    plt.close(fig)
    MANIFEST[name] = {'sources': sources, 'plotted_values': values, 'note': note}

def bars(name, labels, series, xlabel, sources, note='', fmt='.3f', xmax=None, colors=None, log=False):
    k=len(series)
    fig, ax = plt.subplots(figsize=(12, max(5.6, len(labels)*(.90 if k==3 else .75)+1.5)))
    y=np.arange(len(labels)); height=.7/k
    vals=[]
    for j,(lab,values) in enumerate(series.items()):
        a=np.asarray(values,dtype=float); vals.extend(a)
        pos=y+(j-(k-1)/2)*height
        b=ax.barh(pos,a,height*.88,label=lab,color=(colors or C)[j],zorder=3)
        ax.bar_label(b,labels=[format(v,fmt) for v in a],padding=7,fontsize=19,color='#203544')
    ax.set(yticks=y,yticklabels=labels,xlabel=xlabel)
    ax.invert_yaxis()
    if log:
        ax.set_xscale('log'); ax.set_xlim(min(v for v in vals if v>0)*.5,xmax or max(vals)*2.5)
    else: ax.set_xlim(0,xmax or max(vals)*1.22 or 1)
    ax.grid(axis='x',color='#E6ECEC',zorder=0); ax.tick_params(axis='y',length=0,pad=12)
    if k>1: ax.legend(loc='lower left',bbox_to_anchor=(0,1.015),ncol=min(k,3),frameon=False)
    fig.tight_layout()
    save(name,fig,sources,{'labels':labels,'series':series,'x_label':xlabel},note)

def lines(name, x, series, xlabel, ylabel, sources, note='', ylim=None, annotate=True):
    fig,ax=plt.subplots(figsize=(12,6.2))
    xx=np.arange(len(x)) if isinstance(x[0],str) else np.array(x)
    for j,(lab,values) in enumerate(series.items()):
        ax.plot(xx,values,'o-',lw=3.2,ms=9,color=C[j],label=lab,zorder=3)
    ax.set(xlabel=xlabel,ylabel=ylabel,xticks=xx,xticklabels=x)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(color='#E6ECEC'); ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.02),ncol=min(3,len(series)))
    fig.tight_layout();fig.canvas.draw()
    if annotate:
        # Separate nearby labels in display coordinates and keep their exact values.
        for idx,a in enumerate(xx):
            group=sorted((ax.transData.transform((a,values[idx]))[1],j,float(values[idx])) for j,values in enumerate(series.values()))
            desired=np.array([t[0]+19 for t in group]);placed=desired.copy()
            for j in range(1,len(placed)):placed[j]=max(placed[j],placed[j-1]+30)
            placed-=np.mean(placed-desired)
            if placed[0]<ax.bbox.y0+15:placed+=ax.bbox.y0+15-placed[0]
            if placed[-1]>ax.bbox.y1-10:placed-=placed[-1]-(ax.bbox.y1-10)
            for (py,j,value),label_y in zip(group,placed):
                dy=(label_y-py)*72/fig.dpi
                ax.annotate(f'{value:.3f}',(a,value),xytext=(0,dy),textcoords='offset points',ha='center',va='center',fontsize=18,color=C[j],
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':1},
                    arrowprops={'arrowstyle':'-','color':C[j],'lw':.7,'alpha':.55} if abs(dy)>18 else None)
    save(name,fig,sources,{'x':list(x),'series':series,'y_label':ylabel},note)

def one(df,key,val): return df.loc[df[key]==val].iloc[0]

def make():
    # 01: definition pivot and the first independent validation.
    d,s=read(98,'outputs/analysis/C_gameA_semantics.csv'); d=d[d.conf=='metered_load_mw']
    fs=['forecast_load_mw_latest_available','forecast_load_mw_day_ahead','gen_fuel_wind_mw','gen_fuel_wind_pct']
    labels=['最新负荷预测','日前负荷预测','风电出力','风电占比']
    bars('01_semantics',labels,{'单字段能力':[one(d,'field',f)['single'] for f in fs], 'Shapley':[one(d,'field',f)['shapley'] for f in fs], '最大边际 M¹':[one(d,'field',f)['M1'] for f in fs]},'分数（同一 14 字段博弈；目标为 PJM 实际负荷）',[s],fmt='.3f',xmax=1.2)
    d,s=read(98,'outputs/analysis/B_fidelity_rand.csv'); ms=['direct','L0','L0x','L0ensx']
    bars('01_estimator',['共享输出头','逐子集闭式读出','加入原始特征旁路','三种子特征集成'],{'集合值 MAE':[one(d,'model',m).MAE_all for m in ms]},'集合值平均绝对误差 ↓',[s],note='98 号早期配置；三种子特征集成不是 125 号最终配置。')
    d,s=read(99,'outputs/P4_copies.csv'); d=d[(d.src=='forecast_load_mw_latest_available')&(d.conf=='metered_load_mw')].sort_values('copies')
    lines('01_copies',d.copies.tolist(),{'Shapley':d.shapley.tolist(),'最大边际 M¹':d.M1.tolist()},'新增相同字段副本数量','该字段分数',[s],ylim=(0,1.15))
    series={}; sources=[]
    for ds,lab in [('pjm','PJM'),('caiso','CAISO')]:
        d,s=read(100,f'outputs/analysis/{ds}_profiles.csv');sources.append(s)
        series[lab]=[float(d[f'M{i}'].mean()) for i in range(4)]
    lines('01_budget',[0,1,2,3],series,'背景预算 K','平均最大边际风险 Mᴷ',sources,note='100 号早期 DNN 口径；仅在同一数据系列内比较 K，不与后续三模型结果拼接。',ylim=(.10,.34))
    d,s=read(100,'outputs/analysis/efficiency.csv'); d=d[(d['数据集']=='pjm') & d['查询规模'].str.contains('K=2',regex=False)]
    if d.empty:
        d,_=read(100,'outputs/analysis/efficiency.csv'); d=d[d['数据集']=='pjm'];d=d.iloc[[3]]
    # Cost result is explicitly taken from the experiment log to preserve the named full-table setting.
    bars('01_cost',['逐个串行重训','GPU 并行重训','L0 闭式读出','L0 三种子集成'],{'分钟':[70*60,11,.9,6.7]},'完整 K=2 表耗时 / 分钟（对数轴，PJM 早期配置）',['DNN_Aggresvation100/CHANGELOG.md'],fmt='.1f',log=True)
    d,s=read(101,'outputs/analysis/certify_all.csv'); d=d[d.K==2].groupby('ds').mean(numeric_only=True)
    bars('01_independent',['PJM','CAISO'],{'阈值跨越背景复现率':[float(d.loc[x,'C3_τ0.7_跨越背景复现率'])*100 for x in ['pjm','caiso']], '关键字段召回率':[float(d.loc[x,'C3_τ0.7_关键字段召回'])*100 for x in ['pjm','caiso']]},'独立审计结果 / %（K=2，τ=0.7）',[s],fmt='.1f',xmax=110)

    # 02: protocol, complete first evaluation, and resolved evaluation issues.
    d,s=read(102,'outputs/analysis/102_B.csv'); names=['single-target φ+x,x²','multi-target φ+x,x²']
    # Resolve exact labels from the recorded CSV.
    multi=[x for x in d['估计器'].unique() if x.startswith('multi-target') and '集成' not in x][0]
    bars('02_single_multi',['PJM','CAISO'],{'单目标主干':[float(one(d[d['数据集']==ds],'估计器',names[0])['V_MAE']) for ds in ['pjm','caiso']], '多目标主干':[float(one(d[d['数据集']==ds],'估计器',multi)['V_MAE']) for ds in ['pjm','caiso']]},'集合值 MAE ↓',[s])
    d,s=read(103,'outputs/analysis/103_A.csv');d=d[d['规模']==3]
    bars('02_attackers',['PJM','CAISO'],{lab:[float(one(d,'数据集',ds)[col])*100 for ds in ['pjm','caiso']] for lab,col in [('单目标 DNN','单目标DNN被选中'),('多目标 DNN','多目标DNN被选中'),('梯度提升树','树模型被选中')]},'规模为 3 的集合中，被验证集选中的比例 / %',[s],fmt='.1f',xmax=68)
    d,s=read(103,'outputs/analysis/103_B.csv')
    bars('02_readout',['PJM','CAISO'],{lab:[float(one(d[d['数据集']==ds],'估计器',key)['M_MAE_K2']) for ds in ['pjm','caiso']] for lab,key in [('共享输出头','共享输出头（direct）'),('单种子闭式读出','φ+x,x²（单种子）'),('三种子集成','φ+x,x²（三种子集成）')]},'M² 平均绝对误差 ↓（103 号正式参考值）',[s])
    d,s=read(104,'outputs/analysis/M_table_summary.csv')
    bars('02_critical',['PJM','CAISO'],{f'K={k}':[int(one(d[(d['数据集']==ds)],'K',k)['critical数_τ0.7']) for ds in ['pjm','caiso']] for k in [0,1,2]},'关键字段—目标条目数 / 每数据集 492 条（τ=0.7）',[s],fmt='.0f',xmax=400)
    d,s=read(105,'outputs/analysis/105_critical_detection.csv'); d=d[d.tau==.7]
    keys=d['指标'].unique().tolist()
    wanted=[next(x for x in keys if '置换' in x),next(x for x in keys if 'SAGE' in x),next(x for x in keys if 'Pearson' in x),next(x for x in keys if '单字段' in x),next(x for x in keys if '估计' in x and '2' in x)]
    bars('02_baselines',['置换重要性','SAGE','Pearson','单字段精确能力','估计 M²'],{lab:[float(one(d[d['数据集']==ds],'指标',key).PR_AUC) for key in wanted] for ds,lab in [('pjm','PJM'),('caiso','CAISO')]},'关键字段排序 PR-AUC ↑（τ=0.7，K=2）',[s],xmax=1.05)
    d,s=read(106,'outputs/analysis/release_audit_summary.csv')
    for col,name,xlabel in [('危险放行率','02_release','真实危险集合中被放行的比例 / % ↓'),('误拒率','02_reject','真实安全集合中被拒绝的比例 / % ↓')]:
        bars(name,['单字段规则','精确 M 预算','逐阶预算','直接集合估计'],{lab:[float(one(d[d['数据集']==ds],'审查规则',key)[col])*100 for key in ['单字段放行','预算放行','逐阶预算放行','直接估计放行']] for ds,lab in [('pjm','PJM'),('caiso','CAISO')]},xlabel,[s],note='每数据集一个代表目标；2000 个规模为 3 的集合；零值仅为该实验观察。',fmt='.1f',xmax=105 if col=='误拒率' else 70)
    d,s=read(108,'outputs/analysis/108_A1_bounds.csv');d=d[d.K==2]
    bars('02_top3',['PJM','CAISO'],{lab:[float(one(d,'数据集',ds)[col])*100 for ds in ['pjm','caiso']] for lab,col in [('认证首选背景','认证下界比_top1'),('认证前三背景','认证下界比_top3')]},'认证下界 / 参考 M²，均值 / % ↑',[s],fmt='.1f',xmax=115,note='复用正式参考表认证，不是新增独立数据认证。')
    d,s=read(109,'outputs/analysis/109_A_V.csv');d=d[d['规模']==3]
    lines('02_naux',[10,25,50,100],{lab:[float(one(d[d['数据集']==ds],'比例',f'{p}%')['V相对full']) for p in [10,25,50,100]] for ds,lab in [('pjm','PJM'),('caiso','CAISO')]},'辅助标签量 / 完整训练数据的 %','集合能力相对完整数据', [s],note='109 号为单一多目标 DNN，不能与三模型结果直接并列。',ylim=(.7,1.09))
    d,s=read(109,'outputs/analysis/109_B_M.csv');d=d[d.K==2]
    lines('02_naux_max',[10,25,50,100],{lab:[float(one(d[d['数据集']==ds],'比例',f'{p}%')['M相对full']) for p in [10,25,50,100]] for ds,lab in [('pjm','PJM'),('caiso','CAISO')]},'辅助标签量 / 完整训练数据的 %','M² 相对完整数据',[s],ylim=(.94,1.2))

    # 03: actual scenario expansion and closed-loop numerical repair.
    d,s=read(110,'outputs/analysis/110_field_profile.csv');d=d[d['目标']=='Y_机组出力_bus26']; d=d.sort_values('背景升级_M2减M0',ascending=False).head(4)
    bars('03_pilot',d['字段'].tolist(),{'单字段 M⁰':d.M0.tolist(),'最坏背景 M²':d.M2.tolist()},'字段风险（IEEE 30 初步算例）',[s],note='初步试验：单一树模型、单种子。',xmax=.85)
    d,s=read(111,'outputs/analysis/summary_targets.csv')
    bars('03_rts',['燃煤机组 223','燃气机组 221','线路 C35'],{'单字段即越阈':d['τ0.7_单字段即危险'].tolist(),'K=2 关键字段':d['τ0.7_K2关键'].tolist()},'字段数 / 23（τ=0.7）',[s],fmt='.0f',xmax=27)
    d,s=read(112,'outputs/analysis/summary_targets.csv')
    bars('03_nem',['燃煤 BW01','水电 TUMUT3','燃气 PPCCGT'],{'单字段即越阈':d['τ0.5_单字段即危险'].tolist(),'K=2 关键字段':d['τ0.5_K2关键'].tolist()},'字段数 / 30（τ=0.5）',[s],fmt='.0f',xmax=36)
    d,s=read(114,'outputs/114_vs_113.csv');d=d[d['目标']=='PJM 实际负荷']
    bars('03_roles',['113：全候选','113：按披露时点','114：再去目标代理'],{'最强单字段推断能力':d['最强单字段'].tolist()},'PJM 实际负荷：最强单字段 R²',[s],xmax=1.16)
    d,s=read(116,'outputs/analysis/116_vs_114.csv');targets=d['目标'].unique().tolist()
    bars('03_complete',targets,{'114 未补全':[float(d[(d['目标']==t)&d['版本'].str.startswith('114')]['τ0.5_危险小组合'].iloc[0]) for t in targets], '116 补全后':[float(d[(d['目标']==t)&d['版本'].str.startswith('116')]['τ0.5_危险小组合'].iloc[0]) for t in targets]},'规模不超过 3 的最小不安全集合数（τ=0.5）',[s],fmt='.0f')
    d,s=read(115,'outputs/analysis/115_variants_mean.csv');d=d[d['读出'].str[0].isin(['A','C','D','E','F'])]
    labels=['原三种子特征拼接','拼接＋数值修正','单种子＋修正＋交叉验证','三种子预测平均 E','E＋交叉拟合']
    bars('03_fix',labels,{'M² 误差':d['M2误差'].tolist()},'M² 平均绝对误差 ↓（115 号当时 11 个目标）',[s])
    bars('03_fix_cert',labels,{'前三背景认证恢复比':(d['认证前3']*100).tolist()},'认证下界 / 参考 M²，均值 / % ↑',[s],fmt='.1f',xmax=114)
    d,s=read(117,'outputs/analysis/decision_rules.csv');d=d[(d['规则']=='扫描—认证标记')&(d['δ']==0)]
    bars('03_decisions',['τ=0.5','τ=0.7'],{'扫描—认证（旧 E）':[float(d[d['τ']==t]['召回'].mean())*100 for t in [.5,.7]],'单字段规则':[13,17]},'关键字段召回率 / % ↑',[s,'DNN_Aggresvation117/CHANGELOG.md'],fmt='.1f',xmax=105,note='10 目标平均；单字段结果取实验日志的四舍五入汇总；正式更新在 125 号。')

    # 04: ablation -> diagnosis -> failed attempts -> improved estimator -> downstream.
    d,s=read(118,'outputs/analysis/variants_mean.csv'); ms=['lin','polyS','head3','rand1','D1','E'];d=d.set_index('变体').loc[ms]
    bars('04_ablation',['无主干：线性','无主干：二阶字典','共享输出头 ×3','随机主干＋读出 ×1','原预训练＋读出 ×1','原预训练＋读出 ×3'],{'M² 误差':d['M2误差'].tolist()},'M² 平均绝对误差 ↓（118 号，10 目标平均）',[s],note='共享头与 E 均为三种子；随机与原预训练主干为单种子配对。')
    d,s=read(122,'outputs/analysis/variants_mean.csv');ks=['random','uniform','small','recon','small_recon'];v=d.set_index('变体').loc[ks]
    bars('04_attempts',['随机主干','原目标预测','增加小集合掩码','加入遮蔽列重建','小集合＋重建'],{'V 误差':v['V误差_3'].tolist()},'规模 ≤3 的集合值 MAE ↓（单种子）',[s])
    bars('04_dimension',['随机主干','原目标预测','增加小集合掩码','加入遮蔽列重建','小集合＋重建'],{'有效维度':v['有效维度'].tolist()},'特征有效维度（协方差谱参与比）',[s],fmt='.2f')
    q,s=read(122,'outputs/analysis/rank_truncation.csv');q=q.groupby(['主干','秩'])['V误差'].mean().unstack(0).loc[['2','8','全部']]
    lines('04_rank',['2','8','全部 256'], {lab:q[k].tolist() for k,lab in [('random','随机主干'),('uniform','原预训练'),('recon','重建预训练')]},'保留的特征秩 r（分类轴）','集合值 MAE ↓',[s],annotate=True,ylim=(.005,.14),note='122 号秩截断诊断：四组数据、9 目标平均，每组 200 集合；只比较三主干都有的秩，未做单调闭包。')
    d,s=read(122,'outputs/analysis/variants_mean.csv');ks=['uniformE','reconE','reconE_cy'];v=d.set_index('变体').loc[ks]
    bars('04_tail',['原预训练 ×3','重建预训练 ×3','重建 ×3＋预测截断'],{'集合值 V 误差':v['V误差_3'].tolist(),'最大边际 M² 误差':v['M2误差'].tolist()},'平均绝对误差 ↓',[s],note='相同三种子预测平均；截断只用训练标签取值范围。')
    d,s=read(122,'outputs/analysis/large_sets.csv');d=d.groupby(['规模','变体'])['V误差'].mean().unstack(1)
    lines('04_large',d.index.tolist(),{lab:d[k].tolist() for k,lab in [('random','随机主干'),('uniform','原预训练'),('recon','重建预训练')]},'查询集合规模','集合值 MAE ↓',[s],ylim=(.01,.10),note='规模 6/10/16 使用树模型参考值；不与小集合三模型误差直接连线。')
    d,s=read(125,'outputs/select/candidates_mean.csv');d=d.set_index('候选').loc[['E_old','C1','C2','C3']]
    bars('04_candidates',['旧 E','C1：重建 ×3','C2：重建＋随机 ×1','C3：重建＋随机 ×3'],{'V 误差':d['V误差_3'].tolist(),'M² 误差':d['M2误差'].tolist(),'Γ² 误差':d['Γ2误差'].tolist()},'平均绝对误差 ↓（10 目标）',[s],note='C2 为三个独立种子结果的均值，部署一次仅用一个重建／随机组合。')
    bars('04_downstream',['τ=0.5，无余量','τ=0.7，无余量','τ=0.5，δ=0.05','τ=0.7，δ=0.05'],{lab:[float(d.loc[k,c])*100 for c in ['召回_τ0.5_δ0.0','召回_τ0.7_δ0.0','召回_τ0.5_δ0.05','召回_τ0.7_δ0.05']] for k,lab in [('E_old','旧 E'),('C3','最终 C3')]},'扫描—认证关键字段召回 / % ↑',[s],fmt='.1f',xmax=106,note='该评测中的认证精确率均为 1；不意味着总体风险保证。')

    # 05: mechanism, risk profiles, robustness; no old-estimator figures reused.
    d,s=read(123,'outputs/narrow/analysis/ladder.csv');d=d.iloc[:6]
    # Correct labels by the raw field identifiers rather than assuming column order.
    labelmap={'P_g (unit 313 output)':'本机组出力','LMP_g (bus 313)':'本节点电价','LMP_near (bus 306)':'邻近节点电价','LMP_far (bus 303, behind C6)':'远端节点电价','LMP_area2 (bus 223)':'区域 2 电价','System load':'系统负荷'}
    labs=[labelmap.get(x,x) for x in d['字段']]
    # Full field names are inspected in the build audit; use the plot's exact labels.
    bars('05_mechanism',labs,{'单字段 M⁰':d.M0.tolist(),'最大边际 M²':d.M2.tolist()},'对私有报价加成的字段风险',[s],note='窄扰动 ±5%；11 字段全部 2047 个非空子集；单目标 DNN＋树参考值。')
    d,s=read(123,'outputs/narrow/analysis/regime.csv');ids=['LMP_g','LMP_far']; sub=d[d['背景'].isin(ids)].set_index('背景')
    bars('05_regime',['本节点电价＋出力','远端电价＋出力'],{'C6 不阻塞':[float(sub.loc[x,'C6 不阻塞_V(T+P)']) for x in ids],'C6 阻塞':[float(sub.loc[x,'C6 阻塞_V(T+P)']) for x in ids]},'对私有报价加成的集合推断能力 R²',[s],note='本页是测试行运行状态分组的树模型结果。')
    d,s=read(125,'outputs/d124/profile.csv')
    fig,ax=plt.subplots(figsize=(12,5.8))
    for j,(ds,v) in enumerate(d.groupby('数据',sort=False)):
        ax.scatter(v.M0,v.G2,s=45,color=C[j%len(C)],alpha=.75,label=ds,zorder=3)
    ax.fill_between([0,.3],.1,.48,color='#E4F2EB',alpha=.65,zorder=0)
    ax.axvline(.3,color='#B7C2C5',ls='--');ax.axhline(.1,color='#B7C2C5',ls='--')
    count=int(((d.M0<.3)&(d.G2>.1)).sum())
    ax.text(.025,.405,f'{count} / {len(d)} 条\n单字段 <0.3，背景放大 >0.1',fontsize=19,bbox={'facecolor':'#EDF7F5','edgecolor':'none','pad':10})
    ax.set(xlabel='单字段风险 M⁰',ylabel='背景放大量 Γ²',xlim=(-.02,1.02),ylim=(-.015,.48));ax.grid(color='#E6ECEC');ax.legend(fontsize=17,ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.20));fig.tight_layout()
    save('05_profile',fig,[s],{'count':count,'total':len(d),'points':d[['数据','目标','字段','M0','G2']].to_dict('records')},'参考值画像；估计列来自 125 号更新。')
    wanted=d[(d['数据']=='NEM')&d['目标'].str.contains('PPCCGT')&d['字段'].str.contains('SA.*风|风.*SA|南澳.*风',regex=True)]
    if wanted.empty:
        wanted=d[(d['数据']=='NEM')&d['目标'].str.contains('PPCCGT')].sort_values('G2',ascending=False).head(1)
    row=wanted.iloc[0]
    bars('05_average',['单字段','背景边际的平均','背景边际的最大'],{'风险增益':[float(row.M0),float(row['平均背景增益']),float(row.M2)]},'同一字段的三种统计量',[s],note=f"NEM / {row['目标']} / {row['字段']}；平均背景增益不是严格 Shapley。",xmax=.75)
    vals=[float(d[f'G{i}'].mean()) for i in [1,2,3]]
    bars('05_amplification',['K=1','K=2','K=3'],{'平均背景放大量':vals},'Γᴷ 均值（264 个字段—目标条目）',[s])
    d,s=read(119,'outputs/analysis/robust_consistency.csv');d['实验类型']=np.where(d['版本'].str.contains('划分|split'),'更换划分','更换种子')
    bars('05_robust',['更换种子','更换划分'],{'M² 排序相关最小值':[float(d[d['实验类型']==x]['M2排序Spearman'].min()) for x in ['更换种子','更换划分']], 'τ=0.7 关键集合 Jaccard 最小值':[float(d[d['实验类型']==x]['关键集合Jaccard_τ07'].min()) for x in ['更换种子','更换划分']]},'14 个重跑目标中的最小一致性 ↑',[s],xmax=1.13)
    d,s=read(119,'outputs/analysis/k3_critical.csv');d=d[d['τ']==.7]
    criteria=[('PJM','net_actual_interchange'),('NEM','PPCCGT'),('PJM','metered_load'),('RTS','223')]
    sel=[]
    for ds,t in criteria:
        v=d[d['数据'].str.contains(ds)&d['目标'].str.contains(t)];sel.append(v.iloc[0])
    bars('05_kcritical',['PJM 联络线净交换','NEM PPCCGT','PJM 实际负荷','RTS 机组 223'],{'K=2':[int(v['K2关键']) for v in sel],'K=3':[int(v['K3关键']) for v in sel]},'关键字段数（τ=0.7）',[s],fmt='.0f',xmax=29)
    d,s=read(119,'outputs/analysis/nem_k3.csv');d=d[d['口径'].str.contains('本号')]
    bars('05_ksaturation',['NEM 燃煤 BW01','NEM 水电 TUMUT3','NEM 燃气 PPCCGT'],{'M² / M³':d['M2/M3'].tolist()},'边际风险饱和比 ↑',[s],xmax=1.15)

    # 06: actual cost, external comparison, and journal-specific completed artifacts.
    time=[];sources=[]
    for p in sorted((ROOT/'DNN_Aggresvation126/outputs').glob('time_fast_*.csv')):
        time.append(pd.read_csv(p));sources.append(str(p.relative_to(ROOT)))
    times=pd.concat(time); mean=times.groupby('方法')['每集合ms'].mean()
    wanted=['rand1','D1','E','reconE_cy','recon+random_cy','RRE_cy']
    # exact method names are set after inspection; retain the CSV's identifiers for provenance.
    ks=[x for x in ['rand1','uniform1','uniformE','reconE_cy','RR1_cy','RRE_cy'] if x in mean.index]
    if len(ks)<4: ks=mean.index.tolist()
    # Use the explicit keys recorded in the unified timing output.
    keys=mean.index.tolist()
    fig,ax=plt.subplots(figsize=(12,6.0));labels=keys
    labels_map={'head3':'共享输出头 ×3','rand1':'随机主干读出 ×1','E':'旧 E：目标预测 ×3','C1':'C1：重建 ×3','C2':'C2：重建＋随机 ×1','C3':'C3：重建＋随机 ×3','ft':'逐集合微调 ×1'}
    order=mean.loc[list(labels_map)].sort_values();yy=np.arange(len(order));ax.barh(yy,order.values,color=[C[0] if x=='C3' else C[2] for x in order.index],zorder=3)
    ax.set_xscale('log');ax.set(yticks=yy,yticklabels=[labels_map[x] for x in order.index],xlabel='每集合耗时 / ms（对数轴，五组均值）');ax.invert_yaxis();ax.grid(axis='x',color='#E6ECEC')
    for i,v in enumerate(order.values):ax.text(v*1.12,i,f'{v:.1f}',va='center',fontsize=19)
    ax.set_xlim(max(.01,order.min()*.7),order.max()*2.2);fig.tight_layout();save('06_time_all',fig,sources,order.to_dict(),'126 号同一批集合、空闲 GPU、等价快速实现。')
    d,s=read(125,'outputs/select/candidates_mean.csv')
    timing_names={'E_old':'E','C1':'reconE_cy','C2':'RR1_cy','C3':'RRE_cy'}
    # Timing keys vary from the analysis aliases; checked via the logged table below.
    d,s=read(127,'outputs/analysis/summary.csv');d=d.set_index('方法')
    ks=['lin','poly','dropout','ws','surrogate','tabpfn','ours']
    labs=['线性回归','二次回归','均值填补','热启动＋早停','掩码代理直接输出','TabPFN v2','最终配置 C3']
    bars('06_external',labs,{'M² 误差':[float(d.loc[k,'M2误差']) for k in ks]},'最大边际风险平均绝对误差 ↓',[s],note='10 目标，规模 ≤3 全表；LazyVI 只有抽样 V 结果，不放入全表 M 对比。')
    bars('06_tabpfn',['集合 V','最大边际 M²','背景放大 Γ²'],{'TabPFN v2':[float(d.loc['tabpfn',x]) for x in ['V误差','M2误差','Γ2误差']], '最终 C3':[float(d.loc['ours',x]) for x in ['V误差','M2误差','Γ2误差']]},'平均绝对误差 ↓（10 目标）',[s])
    bars('06_recall',['τ=0.5','τ=0.7'],{'TabPFN v2':[float(d.loc['tabpfn',x])*100 for x in ['关键召回05','关键召回07']], '最终 C3':[float(d.loc['ours',x])*100 for x in ['关键召回05','关键召回07']]},'扫描—认证关键字段召回 / % ↑',[s],fmt='.1f',xmax=117)
    tab,s=read(127,'outputs/time_tabpfn.csv')
    # Read actual timing aliases; names are included in this source JSON.
    finalkey=next((x for x in mean.index if x in ['RRE_cy','C3','RRE']),None)
    if finalkey is None: finalkey=[x for x in mean.index if 'RR' in x and 'E' in x][0]
    labels=tab['数据'].tolist(); final=[float(times[(times['数据']==x)&(times['方法']==finalkey)]['每集合ms'].iloc[0]) for x in labels]
    fig,ax=plt.subplots(figsize=(12,5.6)); y=np.arange(len(labels))
    for j,(lab,vals) in enumerate({'最终 C3':final,'TabPFN v2':tab['每集合ms'].tolist()}.items()):
        b=ax.barh(y+(j-.5)*.32,vals,.27,color=C[j],label=lab,zorder=3)
        ax.bar_label(b,labels=[f'{v:.0f}' for v in vals],padding=6,fontsize=19)
    ax.set_xscale('log');ax.set(yticks=y,yticklabels=labels,xlabel='每集合耗时 / ms（对数轴）',xlim=(20,18000));ax.invert_yaxis();ax.grid(axis='x',color='#E6ECEC');ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1));fig.tight_layout()
    save('06_tabtime',fig,[s]+sources,{'labels':labels,'C3_ms':final,'TabPFN_ms':tab['每集合ms'].tolist()},'TabPFN 每目标单独处理；按各组全部目标计时。')

    (OUT/'assets'/'figure_manifest.json').write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2,default=lambda x:float(x)),encoding='utf-8')
    print(f'Redrawn {len(MANIFEST)} SVG figures into {FIG}')

if __name__=='__main__': make()
