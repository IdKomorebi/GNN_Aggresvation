"""Original vector schematics for V2 and its journal adaptations.
Network positions, waveforms and offer steps illustrate relations, not measured topology.
No external artwork or AI-generated bitmap assets are used.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle, Polygon, FancyArrowPatch, Arc
import style as S
S.setup()
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'figures'
CONFIG=json.loads((ROOT/'version.json').read_text()) if (ROOT/'version.json').exists() else {'audience':'general'}

def arrow(ax,p,q,color=S.INK2,rad=0,lw=1):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=10,color=color,lw=lw,connectionstyle=f'arc3,rad={rad}',shrinkA=2,shrinkB=2))

def txt(ax,x,y,t,**kw):
    ax.text(x,y,t,fontsize=kw.pop('fontsize',8.5),ha=kw.pop('ha','center'),va=kw.pop('va','center'),**kw)

def lock(ax,x,y,color=S.ORANGE,k=1):
    ax.add_patch(Arc((x,y+1.2*k),2.8*k,3.2*k,theta1=0,theta2=180,ec=color,lw=1.1))
    ax.add_patch(Rectangle((x-1.8*k,y-1.4*k),3.6*k,2.8*k,fc='white',ec=color,lw=1.1))
    ax.plot(x,y,'.',color=color,ms=2)

def node(ax,x,y,t='',color=S.BLUE,r=1.5):
    ax.add_patch(Circle((x,y),r,fc='white',ec=color,lw=1.2,zorder=4))
    if t:txt(ax,x,y,t,fontsize=8,color=color,zorder=5)

def turbine(ax,x,y):
    ax.plot([x,x],[y-5,y],color=S.INK2,lw=1)
    for a in [90,210,330]:
        a=np.deg2rad(a);ax.plot([x,x+3*np.cos(a)],[y,y+3*np.sin(a)],color=S.AQUA,lw=1.4)

def grid(ax,x,y,k=1):
    pts=np.array([[0,0],[10,6],[20,0],[10,-6]])*k+np.array([x,y])
    for i,j in [(0,1),(1,2),(2,3),(3,0),(1,3)]:ax.plot(pts[[i,j],0],pts[[i,j],1],color=S.MUTED,lw=.9)
    for a,b in pts:node(ax,a,b,r=.65)
    node(ax,x-2*k,y+6*k,'~',r=2*k);ax.plot([x-2*k,x-2*k,x],[y+4*k,y,y],color=S.INK2,lw=.8)
    turbine(ax,x+10*k,y+13*k)
    ax.add_patch(Polygon([[x+18*k,y+5*k],[x+18*k,y+9*k],[x+20*k,y+11*k],[x+22*k,y+9*k],[x+22*k,y+5*k]],fc='white',ec=S.INK2,lw=1))
    ax.plot([x+20*k,x+20*k],[y+5*k,y],color=S.INK2,lw=.8)

def matrix(ax,x,y,w,h,rows=5,cols=6,visible=None,color=S.BLUE):
    for i in range(rows):
        for j in range(cols):
            v=(j in visible) if visible is not None else True
            ax.add_patch(Rectangle((x+j*w/cols,y+i*h/rows),w/cols*.88,h/rows*.83,fc=color if v else '#E7EBEE',ec='none',alpha=.25+.6*((i*3+j*2)%7)/6 if v else 1))

def network(ax,x,y,color=S.BLUE):
    layers=[[y-3,y,y+3],[y-4,y-1.3,y+1.3,y+4],[y-3,y,y+3]]
    for k in range(2):
        for yy in layers[k]:
            for zz in layers[k+1]:ax.plot([x+k*4,x+(k+1)*4],[yy,zz],color=color,alpha=.22,lw=.5,zorder=1)
    for k,ys in enumerate(layers):
        for yy in ys:node(ax,x+k*4,yy,color=color,r=.72)

def fig_framework():
    aud=CONFIG['audience']; fig,ax=plt.subplots(figsize=(S.TEXTW,3.6));fig.subplots_adjust(.03,.03,.97,.94)
    ax.set(xlim=(0,100),ylim=(0,54));ax.axis('off')
    titles=['Operational data','Risk changes with context','Field profile + evidence']
    if aud=='IJCIP':titles=['Infrastructure information','Coupled release channels','Evidence for release review']
    if aud=='JISA':titles=['Data owner and adversary','Background-dependent leakage','Scan, verify and report']
    for x,t in zip([0,35,71],titles):txt(ax,x,53,t,ha='left',weight='bold',fontsize=9)
    for x in [32,68]:ax.plot([x,x],[4,49],color=S.GRID,lw=.9)
    if aud=='JISA':
        matrix(ax,1,32,14,12,visible=[0,2,4]);lock(ax,23,38)
        txt(ax,8,47,'Published fields');txt(ax,23,45,'Target $y$')
        # User icon representing the party holding auxiliary information.
        node(ax,8,24,r=1.5,color=S.INK2);ax.add_patch(Arc((8,18),8,8,theta1=0,theta2=180,ec=S.INK2,lw=1.2))
        txt(ax,20,23,'Auxiliary\nlabelled history',fontsize=8)
        txt(ax,15,10,'Auditor evaluates\nsupervised inferability',color=S.INK2)
    else:
        grid(ax,4,32,.95);lock(ax,20,27,k=.7)
        txt(ax,15,20,'Shared operating state',color=S.INK2)
        t=np.linspace(0,1,60)
        for y,lab,col,k in [(14,'Price',S.BLUE,1),(9,'Wind',S.AQUA,2),(4,'Load',S.ORANGE,3)]:
            txt(ax,1,y,lab,ha='left',fontsize=8);ax.plot(10+t*19,y+.8*np.sin(t*10+k)+.3*np.cos(t*19),color=col,lw=1)
    ax.add_patch(Ellipse((45,39),16,9,fc=S.PALEBLUE,ec=S.BLUE,lw=.8))
    node(ax,42,39,'j');node(ax,48,39,'k');txt(ax,45,47,'Background $T$')
    node(ax,61,39,'i',S.ORANGE,r=2);txt(ax,61,47,'Field $i$');txt(ax,54,39,'+',fontsize=15)
    # A distribution and its maximum express the defining operation.
    xx=np.linspace(36,63,200);yy=4*np.exp(-((xx-42)/4)**2)+1.2*np.exp(-((xx-54)/4)**2)
    ax.fill_between(xx,21,21+yy,color=S.LIGHTBLUE,alpha=.5);ax.plot(xx,21+yy,color=S.BLUE,lw=1)
    ax.plot([63,63],[20,29],color=S.ORANGE,lw=1.2);txt(ax,64,31,'max',fontsize=8)
    arrow(ax,(50,33),(50,28));txt(ax,50,16,'Contextual gain $\Delta_i(T)$')
    txt(ax,50,8,'All $|T|\leq K$, relative to $B$',fontsize=8,color=S.INK2)
    arrow(ax,(29,33),(35,33));arrow(ax,(66,33),(72,33))
    # Profile decomposition, not a checklist inside a box.
    ax.add_patch(Rectangle((74,35),6,4,fc=S.ORANGE));ax.add_patch(Rectangle((80,35),17,4,fc=S.BLUE))
    txt(ax,77,43,'$M^{(0)}$',color=S.ORANGE);txt(ax,89,43,'$\Gamma^{(K)}$',color=S.BLUE)
    ax.plot([74,97],[32,32],color=S.INK2,lw=.8);txt(ax,85,28,'Worst gain $M^{(K)}$')
    for y,w,lab in [(21,20,'1'),(16,17,'2'),(11,14,'3')]:
        txt(ax,73,y,lab,fontsize=8);ax.plot([77,77+w],[y,y],color=S.LIGHTBLUE,lw=2);ax.scatter(77+w,y,s=17,color=S.BLUE)
    txt(ax,85,4,'Top backgrounds → retraining',fontsize=8)
    S.save(fig,OUT/'fig1_framework')

def fig_model():
    fig,ax=plt.subplots(figsize=(S.TEXTW,4.7));fig.subplots_adjust(.03,.03,.97,.95)
    ax.set(xlim=(0,100),ylim=(0,76));ax.axis('off')
    txt(ax,0,75,'(a) Learn a shared representation once',ha='left',weight='bold',fontsize=9.5)
    matrix(ax,2,55,15,12,visible=[0,2,5]);txt(ax,9,70,'Masked history')
    arrow(ax,(19,61),(26,61));network(ax,29,61);txt(ax,33,51,'Trainable network',fontsize=8)
    arrow(ax,(39,61),(48,61));matrix(ax,50,55,14,12,color=S.AQUA)
    txt(ax,57,70,'Reconstruct columns',fontsize=8.5)
    txt(ax,76,62,'+ predict targets $y$',ha='left')
    txt(ax,76,55,'Freeze features $\phi_\\theta$',ha='left',color=S.BLUE)
    ax.plot([0,100],[47,47],color=S.GRID,lw=.8)
    txt(ax,0,44,'(b) Adapt to each field set $S$',ha='left',weight='bold',fontsize=9.5)
    matrix(ax,2,25,12,11,visible=[1,4]);txt(ax,8,21,'$[x_S,m_S]$')
    arrow(ax,(15,31),(24,36));arrow(ax,(15,29),(24,24))
    network(ax,27,36,S.BLUE);txt(ax,31,41,'Reconstruction',fontsize=8)
    network(ax,27,23,S.AQUA);txt(ax,31,17,'Untrained',fontsize=8)
    arrow(ax,(37,36),(47,32),S.BLUE);arrow(ax,(37,23),(47,28),S.AQUA)
    matrix(ax,48,25,5,10,cols=2,color=S.ORANGE);matrix(ax,53,25,7,10,cols=3,color=S.BLUE);matrix(ax,60,25,7,10,cols=3,color=S.AQUA)
    txt(ax,57,39,'Concatenate features',fontsize=8.5)
    txt(ax,57,21,'$[x_S,x_S^2,\phi_\\theta,\phi_\\vartheta]$',fontsize=8)
    arrow(ax,(69,30),(76,30));ax.add_patch(Rectangle((78,26),8,8,fc=S.PALEBLUE,ec=S.BLUE,lw=.8));txt(ax,82,30,'$\\beta_r$',fontsize=12)
    txt(ax,83,39,'Per-set ridge',fontsize=8.5);arrow(ax,(88,30),(96,30));txt(ax,97,25,'$\hat y_r$')
    txt(ax,77,17,'Repeat for 3 seeds; average predictions',fontsize=8)
    ax.plot([0,100],[13,13],color=S.GRID,lw=.8)
    txt(ax,0,8,'(c)',ha='left',weight='bold',fontsize=9.5)
    txt(ax,18,8,'Scan backgrounds');arrow(ax,(33,8),(41,8));txt(ax,56,8,'Retrain top candidates');arrow(ax,(73,8),(81,8));txt(ax,93,8,'Verify gains')
    S.save(fig,OUT/'fig3_model')

def fig_dispatch():
    fig=plt.figure(figsize=(S.TEXTW,3.2));gs=fig.add_gridspec(1,2,width_ratios=[1.05,1],wspace=.32)
    ax=fig.add_subplot(gs[0]);ax.set(xlim=(0,50),ylim=(0,44));ax.axis('off')
    txt(ax,0,43,'(a) Price information follows the grid',ha='left',weight='bold',fontsize=9)
    # Information relation only: draw bus locations without claiming branch topology.
    ax.plot([7,23,40],[24,24,24],color=S.MUTED,lw=1.4)
    ax.plot([7,7],[24,15],color=S.INK2,lw=1.1);node(ax,7,12,'~',r=3)
    for x,c in [(7,S.BLUE),(23,S.BLUE),(40,S.ORANGE)]:
        ax.plot([x,x],[21,27],color=c,lw=2)
    txt(ax,7,32,'Own bus');txt(ax,23,32,'Nearby');txt(ax,40,32,'Remote')
    txt(ax,7,36,'$\lambda_g$',color=S.BLUE);txt(ax,23,36,'$\lambda_{near}$',color=S.BLUE);txt(ax,40,36,'$\lambda_{far}$',color=S.ORANGE)
    ax.plot([30,33],[24,27],color=S.ORANGE,lw=1.5);ax.plot([30,33],[21,24],color=S.ORANGE,lw=1.5)
    txt(ax,32,17,'Congestion\nchanges price relation',fontsize=8,color=S.ORANGE)
    txt(ax,7,5,'Unit 313\noutput $P_g$',fontsize=8.5);lock(ax,17,11,k=.7);txt(ax,26,8,'Private markup $\kappa$',fontsize=8)
    ax=fig.add_subplot(gs[1]);
    x=np.array([0,65,135,210,285,355]);base=np.array([19,23,28,34,40,40]);offer=base*1.05
    ax.step(x,base,where='post',lw=1.4,color=S.MUTED,label='base offer $c_s$')
    ax.step(x,offer,where='post',lw=1.8,color=S.BLUE,label='private offer $(1+\kappa)c_s$')
    pg=245;lam=34*1.05
    ax.plot([pg,pg],[15,lam],color=S.ORANGE,ls='--',lw=1);ax.plot([0,pg],[lam,lam],color=S.ORANGE,ls='--',lw=1);ax.scatter(pg,lam,color=S.ORANGE,s=30,zorder=5)
    ax.set(xlim=(0,360),ylim=(15,46),xlabel='Unit output $P_g$ (MW)',ylabel='Offer / nodal price (illustrative)')
    ax.set_yticks([]);ax.set_xticks([0,pg,355]);ax.set_xticklabels(['0','$P_g$','355'])
    ax.text(.42,.08,'$\kappa=\lambda_g/c_s(P_g)-1$',transform=ax.transAxes,ha='center',fontsize=10)
    ax.legend(loc='upper left',fontsize=7.5);S.panel(ax,'b','Output identifies the segment',x=-.06,y=1.05)
    S.save(fig,OUT/'fig_dispatch_mechanism')

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    fig_framework();fig_model();fig_dispatch()
