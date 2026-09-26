"""Vector journal graphics: 6.55-inch canvas; embedded PDF fonts; editable SVG text."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
INK, INK2, MUTED, GRID = '#172B3A', '#425563', '#81909B', '#E4E9ED'
BLUE, ORANGE, AQUA, VIOLET = '#0072B2', '#D55E00', '#009E73', '#8064A2'
YELLOW, MAGENTA, GREEN, RED = '#E69F00', '#CC79A7', '#009E73', '#BE3D3D'
LIGHTBLUE, PALEBLUE = '#76B4D6', '#EAF3F8'
TEXTW=6.55

def setup():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Liberation Sans','DejaVu Sans'],
        'mathtext.fontset':'dejavusans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
        'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,'legend.frameon':False,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
        'axes.edgecolor':INK2,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK2,'ytick.color':INK2,
        'xtick.major.size':3,'ytick.major.size':3,'grid.color':GRID,'grid.linewidth':.6,
        'figure.facecolor':'white','axes.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','savefig.bbox':None,'lines.linewidth':1.5})

def panel(ax, letter, title='', x=0, y=1.05):
    ax.text(x,y,f'({letter})  {title}',transform=ax.transAxes,va='bottom',weight='bold',fontsize=9.5)

def ygrid(ax):
    ax.grid(axis='y');ax.set_axisbelow(True)

def save(fig,path):
    from matplotlib.text import Text
    for text in fig.findobj(Text):
        if text.get_fontsize() < 7.4: text.set_fontsize(7.4)
    fig.savefig(str(path)+'.pdf',bbox_inches='tight',pad_inches=.10,metadata={'Creator':'Codex V2 editorial revision of Claude V6; original experiment outputs'})
    fig.savefig(str(path)+'.svg',bbox_inches='tight',pad_inches=.10)
    fig.savefig(str(path)+'.png',dpi=180,bbox_inches='tight',pad_inches=.10)
    plt.close(fig)

GRAYS=['#B9C3CA','#95A3AD','#707F89','#425563']
def heatmap(ax, matrix, cmap):
    import numpy as np
    rows,cols=matrix.shape
    mesh=ax.pcolormesh(np.arange(cols+1)-.5,np.arange(rows+1)-.5,matrix,cmap=cmap,vmin=0,vmax=1,shading='flat',rasterized=False)
    ax.set_ylim(rows-.5,-.5)
    return mesh
