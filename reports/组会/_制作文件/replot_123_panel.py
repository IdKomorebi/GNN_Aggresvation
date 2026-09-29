"""Separate panel (b), whose original labels touch panel (c).

Same stored experimental values and definition as the original Matplotlib plot.
No training, smoothing, extrapolation, or new experimental observations.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'DNN_Aggresvation123/outputs/narrow/analysis/regime.csv'
OUT=ROOT/'reports/组会/05_机理画像与稳健性验证/V1/素材'
OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(SOURCE).set_index('背景')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,
 'axes.spines.top':False,'axes.spines.right':False,
 'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
fig,ax=plt.subplots(figsize=(6.4,4.8),layout='constrained')
x=np.arange(2); width=.33
for i,(state,label,color) in enumerate([
 ('C6 不阻塞','uncongested','#9FC5ED'),('C6 阻塞','C6 congested','#2B7ADB')]):
    values=[df.loc[b,state+'_V(T+P)']-df.loc['∅',state+'_V(T+P)'] for b in ['LMP_g','LMP_far']]
    bars=ax.bar(x+(i-.5)*width,values,width,label=label,color=color,zorder=3)
    for bar,v in zip(bars,values):
        ax.annotate(f'{v:.3f}',(bar.get_x()+bar.get_width()/2,v),
            xytext=(0,5 if v>=0 else -17),textcoords='offset points',ha='center',fontsize=12)
ax.set_xticks(x,['own-bus LMP','remote LMP'])
ax.set_ylabel('amplification of $P_g$')
ax.axhline(0,color='#666666',linewidth=.8)
ax.grid(axis='y',color='#DDDDDD',linewidth=.8,zorder=0)
ax.set_ylim(-.034,.325);ax.legend(frameon=False,loc='upper right',fontsize=11)
fig.savefig(OUT/'123_运行状态子图.png',dpi=220)
plt.close(fig)
print(OUT/'123_运行状态子图.png')
