"""Vector journal graphics: 7.16-inch canvas; embedded PDF fonts; editable SVG text."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
INK, INK2, MUTED, GRID = '#172B3A', '#425563', '#81909B', '#E4E9ED'
BLUE, ORANGE, AQUA, VIOLET = '#0072B2', '#D55E00', '#009E73', '#8064A2'
YELLOW, MAGENTA, GREEN, RED = '#E69F00', '#CC79A7', '#009E73', '#BE3D3D'
LIGHTBLUE, PALEBLUE = '#76B4D6', '#EAF3F8'
TEXTW=7.16

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
    fig.savefig(str(path)+'.pdf',metadata={'Creator':'Codex V1 editorial revision; figures regenerated from V4 experiment outputs'})
    fig.savefig(str(path)+'.svg')
    fig.savefig(str(path)+'.png',dpi=180)
    plt.close(fig)
