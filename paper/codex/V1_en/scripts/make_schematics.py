"""Original vector mechanism drawings for the editorial revision of Claude V4_en.
Grid topology, waveforms, matrices and node positions are illustrative, not observations.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle, Polygon, FancyArrowPatch, Arc
import style as S
S.setup()
OUT=Path(__file__).resolve().parents[1]/'figures'

def arrow(ax,p,q,color=S.INK2,lw=1,rad=0):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=9,color=color,lw=lw,connectionstyle=f'arc3,rad={rad}',shrinkA=2,shrinkB=2))

def node(ax,x,y,label,color=S.BLUE,r=2):
    ax.add_patch(Circle((x,y),r,facecolor='white',edgecolor=color,lw=1.3,zorder=3))
    ax.text(x,y,label,ha='center',va='center',fontsize=9,color=color,zorder=4)

def turbine(ax,x,y,scale=1):
    ax.plot([x,x],[y-5*scale,y],color=S.INK2,lw=1.2)
    for a in [90,210,330]:
        a=np.deg2rad(a);ax.plot([x,x+3.3*scale*np.cos(a)],[y,y+3.3*scale*np.sin(a)],color=S.AQUA,lw=1.5)
    ax.add_patch(Circle((x,y),.4*scale,fc=S.AQUA,ec='none'))

def house(ax,x,y,scale=1):
    ax.add_patch(Polygon([(x-2*scale,y),(x-2*scale,y+3*scale),(x,y+5*scale),(x+2*scale,y+3*scale),(x+2*scale,y)],closed=True,fc='white',ec=S.INK2,lw=1))
    ax.add_patch(Rectangle((x-.5*scale,y),scale,2*scale,fc=S.PALEBLUE,ec=S.INK2,lw=.7))

def lock(ax,x,y,color=S.BLUE,scale=1):
    ax.add_patch(Arc((x,y+1.2*scale),2.3*scale,3*scale,theta1=0,theta2=180,ec=color,lw=1.3))
    ax.add_patch(Rectangle((x-1.7*scale,y-1.1*scale),3.4*scale,2.5*scale,fc='white',ec=color,lw=1.2))
    ax.plot(x,y,'.',color=color,ms=2)

def fig_framework():
    fig,ax=plt.subplots(figsize=(S.TEXTW,3.8));fig.subplots_adjust(left=.025,right=.975,bottom=.055,top=.94)
    ax.set(xlim=(0,100),ylim=(0,49));ax.axis('off')
    for x,l in [(0,'(a) Operational fields'),(35,'(b) Contribution in context'),(72,'(c) Auditable field risk')]:
        ax.text(x,48,l,fontsize=9.5,weight='bold',va='top')
    for x in [32,69]:ax.plot([x,x],[3,45],color=S.GRID,lw=.8)
    # A small single-line grid, with actual generator/load symbols.
    pts=[(5,31),(15,37),(26,31),(15,25)]
    for i,j in [(0,1),(1,2),(2,3),(3,0),(1,3)]:
        ax.plot([pts[i][0],pts[j][0]],[pts[i][1],pts[j][1]],color=S.MUTED,lw=1.1)
    for x,y in pts:ax.add_patch(Circle((x,y),.65,fc=S.BLUE,ec='white',lw=.5))
    node(ax,3,37,'~',r=2);ax.plot([3,3,5],[35,31,31],color=S.INK2,lw=.8)
    turbine(ax,15,42,.75);ax.plot([15,15],[38.2,37],color=S.INK2,lw=.8)
    house(ax,26,36,.85);ax.plot([26,26],[36,31],color=S.INK2,lw=.8)
    lock(ax,18.5,28,color=S.ORANGE,scale=.65)
    ax.text(15,21,'Shared operating state',ha='center',fontsize=8.5,color=S.INK2)
    t=np.linspace(0,1,90)
    for y,label,col,k in [(15,'Price',S.BLUE,1),(10,'Wind',S.AQUA,2),(5,'Load',S.ORANGE,3)]:
        ax.text(0,y,label,va='center',fontsize=8.5)
        wave=np.sin(t*7+k)*.8+np.cos(t*17+k)*.35
        ax.plot(9+t*19,y+wave,color=col,lw=1)
    arrow(ax,(29,30),(35,30))
    # Background is a grouping ellipse, and the reviewed field sits outside it.
    ax.add_patch(Ellipse((44,35),16,9,fc=S.PALEBLUE,ec=S.BLUE,lw=.9))
    node(ax,41,35,'j',r=1.7);node(ax,47,35,'k',r=1.7)
    ax.text(44,42,'Background $T$',ha='center',fontsize=8.5)
    node(ax,61,35,'i',color=S.ORANGE,r=2.1)
    ax.text(61,42,'Release',ha='center',fontsize=8.5)
    ax.text(54,35,'+',ha='center',va='center',fontsize=14)
    arrow(ax,(44,30),(44,25));arrow(ax,(61,30),(61,25),S.ORANGE)
    # Inferability bars convey a difference without invented numeric results.
    ax.plot([36,65],[12,12],color=S.INK2,lw=.7)
    ax.add_patch(Rectangle((39,12),7,6,fc=S.LIGHTBLUE,ec='none'))
    ax.add_patch(Rectangle((55,12),7,12,fc=S.BLUE,ec='none'))
    ax.text(42.5,10,'$B\\cup T$',ha='center',va='top',fontsize=9)
    ax.text(58.5,10,'$B\\cup T\\cup\\{i\\}$',ha='center',va='top',fontsize=8.5)
    ax.plot([46,64],[18,18],ls=':',color=S.MUTED,lw=.8)
    ax.annotate('',xy=(64,24),xytext=(64,18),arrowprops=dict(arrowstyle='<->',lw=1,color=S.ORANGE))
    ax.text(65,21,'$\\Delta_i$',ha='left',va='center',fontsize=10,color=S.ORANGE)
    ax.text(50,3,'Maximize over $|T|\\leq K$',ha='center',fontsize=9)
    arrow(ax,(67,30),(72,30))
    # Actual ranking visual and a marked witness, rather than a text card.
    ax.text(84,41,'Scan backgrounds',ha='center',fontsize=9)
    for y,end in [(35,94),(30,90),(25,85)]:
        ax.plot([77,end],[y,y],color=S.LIGHTBLUE,lw=2.4)
        ax.scatter(end,y,s=26,color=S.BLUE,zorder=3)
    ax.text(74,35,'1',va='center',ha='center');ax.text(74,30,'2',va='center',ha='center');ax.text(74,25,'3',va='center',ha='center')
    ax.plot([96,97.2,99],[35,33.8,36.2],color=S.AQUA,lw=1.7)
    ax.text(84,20,'Retrain selected witnesses',ha='center',fontsize=8.5)
    arrow(ax,(85,18),(85,13),S.AQUA)
    for x,l,c in [(77,'j',S.BLUE),(84,'k',S.BLUE),(93,'i',S.ORANGE)]:node(ax,x,8,l,c,r=1.8)
    ax.add_patch(Ellipse((84.5,8),23,7,fill=False,ec=S.ORANGE,lw=1,ls='--'))
    ax.text(85,1.5,'Flag fields / break unsafe sets',ha='center',fontsize=8.5)
    S.save(fig,OUT/'fig1_framework')

def matrix(ax,x,y,w,h,rows=6,cols=6,mask=None,color=S.BLUE):
    rng=np.random.default_rng(7);v=rng.uniform(.25,.9,(rows,cols))
    for i in range(rows):
        for j in range(cols):
            hidden=mask is not None and not mask[i,j]
            ax.add_patch(Rectangle((x+j*w/cols,y+i*h/rows),w/cols-.18,h/rows-.18,fc=S.GRID if hidden else color,alpha=1 if hidden else v[i,j],ec='none'))

def network(ax,x,y):
    layers=[(x,[y-4,y,y+4]),(x+6,[y-6,y-2,y+2,y+6]),(x+12,[y-4,y,y+4])]
    for (xa,ya),(xb,yb) in zip(layers,layers[1:]):
        for a in ya:
            for b in yb:ax.plot([xa,xb],[a,b],color='#C5D5E0',lw=.6,zorder=1)
    for xx,ys in layers:
        for yy in ys:ax.add_patch(Circle((xx,yy),.85,fc='white',ec=S.BLUE,lw=1,zorder=2))

def fig_model():
    fig,ax=plt.subplots(figsize=(S.TEXTW,4.1));fig.subplots_adjust(left=.025,right=.975,bottom=.04,top=.95)
    ax.set(xlim=(0,100),ylim=(0,54));ax.axis('off')
    ax.text(0,53,'(a) Learn a conditional representation once',fontsize=9.5,weight='bold',va='top')
    mask=np.random.default_rng(5).random((6,7))>.4
    matrix(ax,3,34,14,11,6,7,mask)
    ax.text(10,48,'Masked records',ha='center',fontsize=9)
    ax.text(10,31,'$[x\\odot m,\\ m]$',ha='center',fontsize=9)
    arrow(ax,(19,40),(28,40))
    network(ax,31,40)
    ax.text(37,48,'Shared MLP',ha='center',fontsize=9)
    ax.text(37,30,'3 layers × 256',ha='center',fontsize=8.5)
    arrow(ax,(45,40),(56,40))
    for i in range(3):
        ax.add_patch(Circle((60,36+i*4),1.2,fc=S.PALEBLUE,ec=S.BLUE,lw=1))
    ax.text(61,48,'Target predictions',ha='center',fontsize=9)
    arrow(ax,(64,40),(74,40))
    ax.text(85,42,'Supervised loss',ha='center',fontsize=9)
    ax.text(85,37,'Training / validation rows',ha='center',fontsize=8.5,color=S.INK2)
    ax.annotate('',xy=(39,33),xytext=(85,33),arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=-.12',color=S.AQUA,lw=1))
    ax.plot([0,100],[26,26],color=S.GRID,lw=.8)
    ax.text(0,24,'(b) Adapt the readout to each field set',fontsize=9.5,weight='bold',va='top')
    ax.text(10,18,'Query mask $m_S$',ha='center',fontsize=9)
    for j in range(7):ax.add_patch(Rectangle((2+j*2.4,12),2.1,2.5,fc=S.BLUE if j in (1,3,5) else S.GRID,ec='none'))
    arrow(ax,(21,13),(27,13))
    matrix(ax,29,7,19,10,6,9,color=S.BLUE)
    # delineate the concatenated feature blocks, not processing boxes
    for xx in [33.2,37.4]:ax.plot([xx,xx],[7,17],color='white',lw=2)
    ax.text(31,4,'$x_S$',ha='center');ax.text(35.3,4,'$x_S^2$',ha='center');ax.text(43,4,'$\\phi_\\theta$',ha='center')
    lock(ax,47,20,scale=.7)
    ax.text(37,20,'Feature matrix $Z_r(S)$',ha='center',fontsize=8.5)
    arrow(ax,(50,13),(57,13))
    for i in range(6):ax.add_patch(Rectangle((59,7+i*1.7),2.2,1.4,fc=S.ORANGE,alpha=.3+.1*i,ec='none'))
    ax.text(63,20,'Fit $\\beta_r(S)$',ha='center',fontsize=9)
    ax.text(63,3,'Ridge, train-only CV',ha='center',fontsize=8.5)
    arrow(ax,(65,13),(73,13))
    ax.plot([76,94],[7,17],color=S.MUTED,lw=.8,ls='--')
    x=np.arange(77,95,2);y=7+(x-76)*.55+np.array([.8,-.5,.4,-.8,.2,-.3,.6,-.4,.3])
    ax.scatter(x,y,s=13,color=S.BLUE)
    ax.text(85,20,'Average 3 seeds',ha='center',fontsize=9)
    ax.text(85,3,'Test $R^2$ → $\\hat V_y(S)$',ha='center',fontsize=9)
    S.save(fig,OUT/'fig3_model')

if __name__=='__main__':
    fig_framework();fig_model()
