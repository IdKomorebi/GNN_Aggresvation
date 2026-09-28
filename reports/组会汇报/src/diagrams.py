"""Editable, code-native vector diagrams for the six research presentations."""
from html import escape

INK='#203544'; TEAL='#147d78'; ORANGE='#d67b3f'; MUTED='#607580'; BLUE='#516d9a'
def text(x,y,s,size=25,color=INK,weight=400,anchor='middle'):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(s)}</text>'
def rect(x,y,w,h,fill='#edf5f2',stroke='#cbdeda',rx=14):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
def line(x1,y1,x2,y2,color=TEAL,arrow=True,dash=False):
    return f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="3"'+(' stroke-dasharray="8 6"' if dash else '')+(' marker-end="url(#arrow)"' if arrow else '')+'/>'
def svg(title,body):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1400 490" role="img" aria-label="{escape(title)}" style="font-family:Noto Sans CJK SC,Microsoft YaHei,sans-serif"><title>{escape(title)}</title><defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto-start-reverse"><path d="M0,1 L8,5 L0,9" fill="none" stroke="{TEAL}" stroke-width="2"/></marker></defs>{body}</svg>'

def flow(title,steps,footer=''):
    n=len(steps);w=min(260,(1310-(n-1)*40)/n);total=n*w+(n-1)*40;x=(1400-total)/2;b=''
    for i,(head,sub) in enumerate(steps):
        xx=x+i*(w+40);b+=rect(xx,120,w,220,'#edf5f2' if i<n-1 else '#e2efeb')
        b+=text(xx+26,158,f'{i+1:02d}',18,TEAL,700,'start')+text(xx+w/2,207,head,27,INK,600)
        for j,t in enumerate(sub.split('|')):b+=text(xx+w/2,257+j*35,t,20,MUTED)
        if i<n-1:b+=line(xx+w+7,230,xx+w+32,230)
    b+=text(700,423,footer,24,TEAL,500)
    return svg(title,b)

def risk():
    b=rect(50,70,400,320,'#f0f4f7','#d4dfe8')+text(250,115,'接收者已有信息',27,BLUE,600)
    for i,(lab,xx) in enumerate([('已公开基底 B',80),('额外背景 T',270)]):
        b+=rect(xx,145,150,175,'white','#ccdce2')+text(xx+75,185,lab,21,BLUE)
        for j in range(3):b+=rect(xx+22,208+j*28,106,16,'#dbe8ed','#dbe8ed',3)
    b+=text(250,367,'仅限制额外背景：|T| ≤ K',23,MUTED)
    b+=rect(515,145,230,175,'#fbefe5','#eccbb2')+text(630,195,'拟评估字段 i',28,ORANGE,600)+text(630,245,'单独看 / 加入背景',22,MUTED)+text(630,280,'同一字段，两种条件',20,MUTED)
    b+=line(457,235,505,235)+line(755,235,837,235)
    b+=rect(852,70,495,320,'white','#d5e4de')+text(1100,117,'对受保护目标 Y 的新增推断能力',26,TEAL,600)
    b+=text(880,182,'已有信息',22,MUTED,400,'start')+rect(1015,160,180,32,'#a7c8c1','#a7c8c1',4)
    b+=text(880,243,'再加字段 i',22,MUTED,400,'start')+rect(1015,220,180,32,'#a7c8c1','#a7c8c1',4)+rect(1198,220,92,32,'#d67b3f','#d67b3f',4)
    b+=text(1243,289,'边际增量 Δ',21,ORANGE,600)+text(1100,351,'遍历预算内背景，取最大增量 M',23,TEAL)
    b+=text(700,449,'示意：比较“加入字段前后”的能力变化；条形长度不代表实验数值',20,MUTED)
    return svg('字段风险的评价对象',b)

def protocol():
    b='';rows=[(70,'训练集','#edf5f2','拟合专用模型','学习共享表示 / 求读出'),(210,'验证集','#edf0f7','选模型、轮次与超参数','不使用测试表现作选择'),(350,'测试集','#fcf0e7','报告样本外 R²','汇总经验能力与风险指标')]
    for y,head,color,a,c in rows:
        b+=rect(45,y,235,105,color)+text(162,y+62,head,30,TEAL,600)
        b+=line(290,y+53,350,y+53)+rect(365,y,485,105,'white')+text(607,y+61,a,27,INK,500)
        b+=line(860,y+53,917,y+53)+rect(930,y,420,105,color)+text(1140,y+61,c,24,MUTED)
    b+=text(700,35,'同一拆分、同一候选集合、同一报告口径',24,TEAL,600)
    return svg('实验数据使用与评价协议',b)

def independent():
    b=rect(45,60,260,355,'#edf5f2')+text(175,130,'独立划分',30,TEAL,600)+text(175,194,'相同字段与目标',22,MUTED)+text(175,240,'数据分成两部分',22,MUTED)+text(175,286,'重复交换角色',22,MUTED)
    b+=line(315,153,405,153)+line(315,329,405,329)
    b+=rect(420,85,375,135,'#edf0f7','#d2daeb')+text(607,137,'发现数据',30,BLUE,600)+text(607,183,'搜索并固定候选背景',24,MUTED)
    b+=rect(420,262,375,135,'#fcf0e7','#eccbb2')+text(607,311,'审计数据',30,ORANGE,600)+text(607,357,'独立重训并评价',24,MUTED)
    b+=line(802,152,894,152)+line(802,329,894,329)
    b+=rect(910,85,440,312,'white')+text(1130,139,'比较三类结果',28,TEAL,600)
    for j,t in enumerate(['风险值是否回落','阈值跨越是否复现','关键字段是否找回']):b+=text(1130,212+j*57,t,26,INK)
    return svg('发现与独立审计的配对验证',b)

def numerical():
    b=flow('数值异常的定位和修复',[('近常数特征','训练标准差极小'),('标准化放大','测试行微小变化|变成大特征值'),('背景能力失真','预测异常|V(T) 被压到 0'),('边际被放大','差分后取最大|选中异常背景')],'').split('</svg>')[0]
    # replace the first row geometry with room for the applied remedies.
    b=b.replace('y="120"','y="40"').replace('y="158"','y="78"').replace('y="207"','y="127"').replace('y="257"','y="177"').replace('y="292"','y="212"').replace('230','150')
    b+=rect(75,310,1250,115,'#e3f1eb','#c2dacc')+text(700,354,'本批次已实施的修复',25,TEAL,700)+text(700,397,'标准差下限  +  特征范围截断  +  训练内选 λ  +  三种子预测平均',25,INK)
    b+=text(700,470,'115 号：训练数据决定防护与调参；专用重训参考值只用于评估',20,MUTED)+'</svg>'
    return b

def estimator():
    b=text(250,40,'一次预学习',24,TEAL,700)+text(900,40,'每个查询子集 S',24,TEAL,700)
    b+=rect(45,85,375,140,'#edf5f2')+text(232,130,'重建主干 φ',29,TEAL,700)+text(232,173,'预测目标＋重建被遮候选列',22,MUTED)+text(232,207,'仅在训练数据上优化',18,MUTED)
    b+=rect(45,275,375,120,'#edf0f7','#ced9e9')+text(232,323,'随机主干 φ̃',29,BLUE,700)+text(232,365,'固定随机初始化，不训练',22,MUTED)
    b+=line(425,150,520,230)+line(425,335,520,260)
    b+=rect(540,115,350,235,'white')+text(715,165,'按可见集合提取特征',27,INK,600)+text(715,218,'[xₛ, xₛ², φ, φ̃]',32,TEAL,600)+text(715,267,'每个种子单独求解读出',23,MUTED)+text(715,310,'训练内交叉验证选 λ',22,MUTED)
    b+=line(895,235,956,235)+rect(975,115,365,235,'#edf5f2')+text(1157,163,'三个种子预测取平均',27,TEAL,600)+text(1157,218,'数值保护与范围截断',23,MUTED)+text(1157,270,'得到 V̂(S)',31,INK,600)+text(1157,312,'再计算 M̂、Γ̂ 与候选背景',21,MUTED)
    b+=text(700,450,'C3 = 重建与随机特征在种子内拼接；种子之间平均预测，共三次读出',24,TEAL,600)
    return svg('125 号最终估计器的训练与查询结构',b)

def mechanism():
    b=text(250,40,'已知机制的受控验证',25,TEAL,700)
    b+=rect(55,92,320,280,'#edf5f2')+text(215,138,'机组 313_CC_1',29,TEAL,600)
    b+='<circle cx="215" cy="210" r="44" fill="white" stroke="#147d78" stroke-width="3"/>'+text(215,222,'G',34,TEAL,600)
    b+=text(215,298,'出力 Pɡ 定位报价段',24,INK)+text(215,341,'私有报价加成 κ',22,ORANGE,600)
    b+=line(375,212,465,212,arrow=False)+'<circle cx="485" cy="212" r="11" fill="#147d78"/>'+text(485,173,'本节点',24,INK)
    b+=line(497,212,865,212,arrow=False)+rect(655,190,60,44,'#fcf0e7','#d67b3f',4)+text(685,175,'C6',23,ORANGE,600)
    b+='<circle cx="885" cy="212" r="11" fill="#516d9a"/>'+text(885,173,'远端节点',24,INK)
    b+=text(590,268,'电价 λɡ',27,TEAL,600)+text(920,268,'远端电价',25,BLUE,600)
    b+=rect(1000,85,335,295,'white')+text(1167,137,'预先写下的预期',26,TEAL,600)
    for j,t in enumerate(['出力＋本节点电价有增益','邻近电价可能替代','阻塞后远端价格失效']):b+=text(1167,201+j*58,t,22,INK)
    b+=rect(65,400,1270,65,'#edf5f2')+text(700,442,'边际机组条件下：λɡ = (1 + κ) · c段(Pɡ)　｜　机制预期在实验前登记',25,INK)
    return svg('受控报价加成算例的机制示意',b)

DIAGRAMS={
 'risk':risk,
 'protocol':protocol,
 'independent':independent,
 'numerical':numerical,
 'estimator':estimator,
 'mechanism':mechanism,
 'workflow':lambda:flow('从集合评估到字段画像',[('任意字段集合','输入可见字段 S'),('通用估计器','快速得到 V̂(S)'),('边际风险计算','M̂、Γ̂、背景排序'),('少量专用验证','固定候选后确认'),('字段风险画像','风险量＋支撑背景')],'本轮增加了字段级产物，并把估计与验证接在同一流程中'),
 'roles':lambda:flow('PJM 与 CAISO 字段口径修订',[('113：重定角色','按披露时点筛选|复用已有参考表'),('114：重新训练','去同目标预测代理|补规模 4 参考值'),('116：补齐候选','纳入遗漏公开字段|全表重新训练')],'完整修订在本次汇报交代：最终采用 116 号场景'),
 'timing':lambda:flow('统一性能测量',[('固定数据与设备','同一空闲 GPU|每组 200 集合'),('统一查询设置','批大小 8|预热后计时'),('等价实现加速','复用 Gram 计算|批量求解 λ'),('核对数值一致','RTS、CAISO 抽查|最大差为 0')],'一次性预训练与每集合查询分别计时；本页不包含新训练实验'),
}
