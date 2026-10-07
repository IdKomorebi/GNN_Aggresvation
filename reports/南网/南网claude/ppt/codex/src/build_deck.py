"""One editable scene description -> native PPT shapes (JS) and individual SVGs."""
from pathlib import Path
from html import escape
from PIL import ImageFont
import json, math, hashlib

ROOT = Path(__file__).resolve().parents[1]
W,H = 1280,720
C = dict(ink='172B3A',muted='647584',blue='2459C4',teal='087F79',amber='AE6C26',
         pale='F4F7FA',bluep='EDF3FD',tealp='EDF7F5',amberp='FBF5E9',line='DCE4EA',white='FFFFFF')
FONT='Noto Sans CJK SC'
FONTS={}
def font(size,bold=False):
    k=(size,bold)
    if k not in FONTS:
        FONTS[k]=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-'+('Bold' if bold else 'Regular')+'.ttc',size,index=2)
    return FONTS[k]
def wrap(text,w,size,bold):
    result=[]
    for para in str(text).split('\n'):
        line=''
        for char in para:
            if line and font(size,bold).getlength(line+char)>w:
                result.append(line);line=char
            else:line+=char
        result.append(line)
    return result

slides=[]
class Page:
    def __init__(self,title,section,subtitle='',source='',appendix=False):
        self.n=len(slides)+1;self.title=title;self.section=section;self.source=source
        self.items=[];self.notes='';self.appendix=appendix;slides.append(self)
        self.text(64,29,1100,section,15,C['blue'],True)
        self.text(64,66,1152,title,38,C['ink'],True,lh=1.22)
        if subtitle:self.text(66,126,1130,subtitle,21,C['muted'],lh=1.3)
        self.line(64,676,1216,676,C['line'],1)
        self.text(64,689,870,'南网项目  /  数据分类分级研究体系',12,C['muted'])
        self.text(1000,687,216,(f'A{self.n-12}' if appendix else f'{self.n:02d} / 12'),14,C['muted'],align='right')
    def rect(self,x,y,w,h,fill=C['white'],stroke=None,r=0,sw=1):
        self.items.append(dict(kind='rect',x=x,y=y,w=w,h=h,fill=fill,stroke=stroke,r=r,sw=sw))
    def circle(self,x,y,d,fill=C['white'],stroke=None,sw=1):
        self.items.append(dict(kind='circle',x=x,y=y,w=d,h=d,fill=fill,stroke=stroke,sw=sw))
    def line(self,x1,y1,x2,y2,color=C['line'],width=2,arrow=False,dash=False):
        self.items.append(dict(kind='line',x=x1,y=y1,x2=x2,y2=y2,color=color,width=width,arrow=arrow,dash=dash))
    def text(self,x,y,w,text,size=24,color=C['ink'],bold=False,align='left',lh=1.4):
        lines=wrap(text,w,size,bold)
        h=len(lines)*size*lh+2
        self.items.append(dict(kind='text',x=x,y=y,w=w,h=h,text='\n'.join(lines),size=size,color=color,bold=bold,align=align,lineHeight=size*lh))
        return h
    def label(self,x,y,w,text,color=C['blue'],fill=C['bluep'],h=36,size=18):
        self.rect(x,y,w,h,fill,r=6)
        self.text(x+8,y+(h-size*1.35)/2,w-16,text,size,color,True,'center',1.35)
    def node(self,x,y,w,h,title,body='',color=C['blue'],fill=None,title_size=26):
        self.rect(x,y,w,h,fill or C['white'],C['line'],r=10)
        self.rect(x,y,4,h,color,r=0)
        self.text(x+22,y+17,w-44,title,title_size,color,True)
        if body:self.text(x+22,y+63,w-44,body,21,C['muted'],lh=1.45)
    def takeaway(self,text,color=C['blue'],y=600):
        self.rect(64,y,1152,48,C['bluep'] if color==C['blue'] else C['tealp'],r=6)
        self.text(82,y+8,1116,text,22,color,True,lh=1.3)
    def cite(self,text):self.text(66,651,1150,text,12,C['muted'],lh=1.25)

# 01 / restrained white cover
p=Page('面向推断风险的\n数据分类分级体系','南方电网项目 · 研究体系汇报')
p.items=[i for i in p.items if not(i['kind']=='text' and i.get('text')==p.title)]
p.text(64,95,1100,'面向推断风险的',44,C['ink'],True)
p.text(64,155,1145,'数据分类分级体系',56,C['ink'],True)
p.text(67,255,1050,'规则确定基线  ·  组合推断补充证据  ·  GRPO 支持联合等级建议',25,C['muted'])
cover=[(64,'规则基线','明确分什么、依据什么定级',C['blue']),
       (464,'推断证据','发现单字段之外的组合影响',C['teal']),
       (864,'等级建议','在硬约束内复核并联合优化',C['blue'])]
for x,title,body,color in cover:
    p.line(x,371,x+340,371,color,3)
    p.text(x,396,340,title,30,color,True)
    p.text(x,452,340,body,23,C['muted'])
for x in (419,819):p.line(x,422,x+29,422,C['muted'],2,True)
p.text(64,574,1110,'形成“有规则依据、有推断证据、可审批、可复评”的分类分级结果。',27,C['ink'],True)
p.notes='本次汇报聚焦分类分级研究体系。先按南网规则形成基线，再检查字段与已有背景结合是否揭示新的受保护信息，最后在规则硬约束内形成等级复核和联合优化建议。GRPO 是重点研究模块，但等级结论仍需业务复核与审批。整套方法统一作为项目组的研究体系介绍。'

# 02 / problem statement, one diagram
p=Page('单项定级之外，还要检查组合后能揭示什么','01  为什么需要推断风险分析',
       '新的风险可能出现在字段之间，也可能来自接收方已经掌握的信息。')
p.text(78,191,455,'单独观察',24,C['muted'],True)
p.text(694,191,470,'联合观察',24,C['teal'],True)
p.line(638,190,638,567,C['line'],1)
for y,name in ((269,'字段 A'),(421,'字段 B')):
    p.node(80,y,165,86,name,color=C['blue'],title_size=25)
    p.line(246,y+43,355,y+43,C['line'],2,True,True)
    p.text(366,y+23,205,'对目标的证据弱',22,C['muted'])
p.node(698,278,152,84,'字段 A',color=C['blue'],title_size=25)
p.node(698,420,152,84,'字段 B',color=C['blue'],title_size=25)
p.line(850,320,906,320,C['teal'],2)
p.line(850,462,906,462,C['teal'],2)
p.line(906,320,906,462,C['teal'],2)
p.line(906,391,959,391,C['teal'],2,True)
p.node(971,335,231,112,'受保护目标','联合后可能恢复',C['teal'],C['tealp'],25)
p.label(710,523,476,'字段集合 + 已有背景 → 推断能力',C['teal'],C['tealp'],42,21)
p.takeaway('需要补充评估：组合后多揭示了什么，以及是否改变原有分级依据。')
p.cite('依据：所供分级手册“深度”要素、附录 D/E；风险评估手册第 382 项。图为原理示意。')
p.notes='规则定级仍然是基础。新增分析解决的是：分别审查每个字段时不明显的信息，可能在组合后被推断出来。这里的目标必须在具体场景中仍需保护；能推断某个已经允许获取的目标，不自动构成分级问题。本页仅示意组合效应，不表示某项南网数据已经发生泄露。'

# 03 / architecture with four outputs, no authorship colors
p=Page('四个阶段，形成有依据、可复核的等级结论','02  总体研究体系',
       '以南网分类分级规则为基线，将推断证据接入等级复核与审批。')
stages=[('01','基础分类分级','规则解析\n业务分类\n影响矩阵','规则等级 + 依据',C['blue']),
        ('02','推断风险分析','主体与已知背景\n集合推断评估\n字段风险解释','推断证据 + 范围',C['teal']),
        ('03','等级复核优化','补充影响判断\nGRPO 联合优化\n硬约束与验证','建议等级 + 理由',C['blue']),
        ('04','审批固化复评','业务复核与审批\n目录固化\n变化触发复评','批准结果 + 记录',C['blue'])]
for k,(num,title,body,out,color) in enumerate(stages):
    x=64+k*296
    p.text(x,208,252,num,46,color,True)
    p.text(x,275,258,title,28,C['ink'],True)
    p.line(x,324,x+249,324,color,3)
    p.text(x,347,251,body,24,C['muted'],lh=1.65)
    p.label(x,492,258,out,color,C['tealp'] if k==1 else C['bluep'],45,20)
    if k<3:p.line(x+261,387,x+283,387,C['muted'],2,True)
p.line(1082,552,1082,578,C['blue'],2)
p.line(1082,578,190,578,C['blue'],2)
p.line(190,578,190,548,C['blue'],2,True)
p.text(451,582,520,'数据、背景或规则变化 → 重新复评',20,C['blue'],align='center')
p.cite('范围：基础分类分级、推断风险增强、等级复核建议、审批固化与动态更新。')
p.notes='每个阶段都有明确产物。第一阶段给规则等级和条款，第二阶段给目标、背景、能力和验证范围，第三阶段将技术证据转成影响复核依据，并通过 GRPO 研究联合等级方案，第四阶段完成审批固化。第二、三阶段不是一次流水线：建议形成后，还要验证其剩余可得组合，必要时返回更新。'

# 04 / classification and grading are distinct
p=Page('基础定级：先分类，再按影响形成规则等级','03  规则基线',
       '大模型与检索增强辅助条款提取，定级依据仍是业务语义和南网规则。')
p.label(64,193,124,'数据分类')
p.text(212,194,990,'确定字段属于哪一类业务对象',25,C['ink'],True)
labels=[('专业域',140),('专业级流程组',235),('操作级流程',205),('业务对象',185),('字段',137)]
x=64
for idx,(lab,width) in enumerate(labels):
    p.rect(x,250,width,68,C['pale'],r=7)
    p.text(x+10,269,width-20,lab,23,C['ink'],True,'center')
    if idx<4:p.line(x+width+9,284,x+width+33,284,C['muted'],2,True)
    x+=width+45
p.label(64,362,124,'数据分级')
p.text(212,363,990,'确定受损可能影响谁、影响多严重',25,C['ink'],True)
for x,w,title,sub in [(64,317,'八个分级要素','含规模、精度、深度等'),(444,360,'影响对象 × 影响程度','按适用规则从严判断'),(867,349,'规则等级 L_rule','保留条款与复核记录')]:
    p.node(x,416,w,114,title,sub,C['blue'],title_size=25)
for x in (389,812):p.line(x,473,x+44,473,C['blue'],2,True)
p.label(64,565,670,'一般数据：1 · 2 · 3 级',C['blue'],C['bluep'],45,24)
p.label(752,565,224,'重要：4 级',C['amber'],C['amberp'],45,23)
p.label(994,565,222,'核心：5 级',C['amber'],C['amberp'],45,23)
p.cite('依据：所供分级手册第四部分、表 4.4、附录 A–C；重要/核心数据按对应识别、申报与审批程序处理。')
p.notes='分类和分级是两件事。分类沿专业域、流程、业务对象定位到字段；分级依据要素和影响矩阵。大模型提取规则、匹配业务语义，输出可审查候选，不能替代影响判断。规则等级是后续研究的基线；如果对象已有批准等级，还需保留现行结果与变更状态。材料为所供版本，正式采用前核对生效状态。'

# 05 / explicitly non-nested recipient reality and retained history
p=Page('场景定义：谁在什么时候已经知道什么','04  推断分析的边界',
       '同一数据，在不同主体、时点和已有背景下，可能形成不同的推断证据。')
p.rect(64,197,486,350,C['pale'],r=10)
p.text(88,218,430,'一张场景卡，固定评估问题',26,C['ink'],True)
for i,(left,right) in enumerate([('接收主体','角色、授权与用途'),('评估时点','披露前 / 披露后'),('已知背景 B','公开、自有、历史已获取'),('受保护目标 y','此时对该主体仍需保护')]):
    y=283+i*61
    p.text(88,y,135,left,22,C['muted'])
    p.text(244,y,281,right,22,C['ink'],True)
p.text(610,205,591,'把“已有”与“可能新增”分开',27,C['ink'],True)
p.node(610,270,605,110,'已有背景 B','已掌握的信息保留，不因重新定级而消失',C['teal'],C['tealp'],26)
p.node(610,403,605,110,'可得字段 H','依据主体、时点、授权及确认的等级映射确定',C['blue'],C['bluep'],26)
p.text(612,544,584,'披露时点改变 → 重新确认 B、H 和目标',23,C['muted'])
p.takeaway('市场“公开信息”面向有关市场成员；不能等同于向社会公众公开。')
p.cite('依据：所供披露细则 2.7、5.1；开放目录说明、注册细则；等级—可见范围映射须经业务确认。')
p.notes='先确定具体主体和时点，再定义它已经持有的信息 B、还能取得的字段 H，以及对它仍需保护的目标 y。已知背景包括历史提供、自有数据和其他合法渠道信息，不仅是网上公开内容。公众、有关市场成员、特定成员的权限并不完全嵌套。实验中按等级阈值生成可见范围，是一种须确认的简化模型；不能直接替代实际授权关系。'

# 06 / two parallel inputs, no association-gate error
p=Page('关联找线索，联合验证确认能否推断','05  从关系线索到推断证据',
       '两路分析相互补充：关联图解释关系，集合评估直接检验联合信息。')
p.rect(64,199,493,339,C['pale'],r=9)
p.rect(592,199,624,339,C['tealp'],r=9)
p.text(87,220,450,'关联发现',28,C['blue'],True)
p.text(615,220,575,'联合推断评估',28,C['teal'],True)
for i,(a,b) in enumerate([('Pearson','线性关系'),('Spearman','单调关系'),('MIC','非线性依赖')]):
    y=281+i*54
    p.text(88,y,183,a,22,C['blue'],True)
    p.text(282,y,224,b,23,C['ink'])
p.text(88,458,439,'产物：候选关联图、方向线索\n方向需结合业务或预测验证',20,C['muted'],lh=1.55)
p.label(617,286,238,'背景 B + 字段集合 S',C['teal'],C['white'],50,22)
p.line(864,310,892,310,C['teal'],2,True)
p.label(905,286,283,'目标专用推断模型',C['teal'],C['white'],50,22)
p.text(620,371,565,'连续目标：样本外预测能力\n结构目标：匹配任务的恢复指标',24,C['ink'],lh=1.6)
p.text(620,472,570,'产物：集合能力 V(B ∪ S) 与可核验结果',23,C['teal'],True)
p.line(310,542,310,570,C['blue'],2)
p.line(904,542,904,570,C['teal'],2)
p.line(310,570,904,570,C['muted'],2)
p.takeaway('关联弱的字段也可能参与强组合：联合评估不能被两两关联图直接排除。',C['teal'])
p.cite('方法定位：关联图提供解释与优先级；模型预测检验推断能力；连续与结构目标分别验证。')
p.notes='关联发现有实际作用：提示副本、提供业务解释、帮助安排验证顺序。但 Pearson、Spearman、MIC 都不是已确认的推断方向，也不直接给出安全等级。联合推断评估回答给定字段集合能恢复目标到何种程度。连续目标可以使用 DNN 等模型，结构目标可研究 GAT 与物理约束，但指标和验证任务要匹配。集合评估应保留图外组合，避免漏掉两两弱、联合强的情形。'

# 07 / one illustrative example for every risk quantity
p=Page('V 判断暴露，M 与 Γ 解释字段为何需要复核','06  风险如何量化',
       '同一目标、同一场景：比较加入字段 i 前后，推断能力增加了多少。')
plotx,plotw=224,534
p.text(66,186,705,'V：在留出数据上的推断能力（示意 R²）',21,C['muted'])
vals=[('已有 B',.10,C['line']),('B + i',.15,C['blue']),('B + T*',.20,C['line']),('B + T* + i',.85,C['teal'])]
for j,(label,value,col) in enumerate(vals):
    y=244+j*78
    p.text(66,y+6,149,label,21,C['ink'])
    p.rect(plotx,y,plotw,42,C['pale'],r=4)
    p.rect(plotx,y,plotw*value,42,col,r=4)
    p.text(plotx+plotw*value+12,y+5,86,f'{value:.2f}',22,C['ink'],True)
tx=plotx+plotw*.7
p.line(tx,226,tx,528,C['amber'],2,False,True)
p.text(tx-112,534,260,'示意复核线 τ = 0.70',19,C['amber'],align='center')
p.line(805,207,805,572,C['line'],1)
for y,tag,val,sub in [(224,'M⁽⁰⁾','0.05','无额外背景时的增量'),(341,'M⁽ᴷ⁾','0.65','已评估背景中的最大增量'),(458,'Γ⁽ᴷ⁾','0.60','背景使字段增益额外放大')]:
    p.text(842,y,120,tag,31,C['teal'],True)
    p.text(1024,y-1,172,val,36,C['teal'],True,align='right')
    p.text(842,y+50,355,sub,21,C['muted'])
p.takeaway('最终能力是否越线，用 V 判断；为什么这个字段值得复核，用 M、Γ 和见证背景解释。',C['teal'])
p.cite('全部数值为原理示意：T* 为示意背景族中的最大见证；τ 需按目标确认；分数不直接映射安全等级。')
p.notes='四条柱形来自同一场景。无额外背景时，加入 i 使能力从 0.10 到 0.15，增量为 0.05。有背景 T* 时，能力从 0.20 到 0.85，增量为 0.65；如果 T* 是已评估背景族的最大见证，则 M 的估计最大值为 0.65，背景放大量为 0.60。0.85 越过示意阈值 0.70，所以需要复核。不能只看增量，因为背景本身已经暴露时，增量很小仍可能有较高总暴露。所有数值仅用于说明原理。'

# 08 / regulatory bridge must precede optimization
p=Page('等级复核：先把推断证据转成影响依据','07  风险证据怎样进入定级',
       '技术越线触发复核；建议等级仍需要回答“影响谁、影响多严重”。')
blocks=[(64,238,266,'推断证据','目标、场景、背景\n能力、验证范围',C['teal']),
        (378,238,266,'影响复核','是否揭示更深信息\n对象与后果是否改变',C['blue']),
        (692,238,266,'规则判定','核对要素与矩阵\n形成允许的候选级别',C['blue']),
        (1006,238,210,'等级建议','附理由与证据\n进入联合优化',C['blue'])]
for x,y,w,title,body,col in blocks:p.node(x,y,w,194,title,body,col,C['tealp'] if col==C['teal'] else C['white'],25)
for x in (338,652,966):p.line(x,333,x+30,333,C['muted'],2,True)
p.label(65,473,365,'规则基线：原则上不低于基线',C['blue'],C['bluep'],52,22)
p.label(447,473,375,'重要 / 核心：按识别程序处理',C['amber'],C['amberp'],52,22)
p.label(839,473,377,'证据不足或约束冲突：专项复核',C['muted'],C['pale'],52,21)
p.text(66,558,1139,'数据项、数据集、衍生产品分别复核；组合的风险不自动复制为每个成员的等级。',23,C['muted'])
p.takeaway('优化只在有依据的候选级别和可行约束内选择方案，不以代价替代影响判断。')
p.cite('依据：所供分级手册表 4.4、附录 D/E；输出区分规则等级、建议等级与最终批准等级。')
p.notes='这是技术与制度之间最重要的接口。经验推断越线说明值得复核，不直接证明某个字段必须继承目标等级。先确认目标在该场景需要保护，再看新的深度、覆盖范围和影响后果是否改变分级要素，然后按矩阵和适用规则形成候选级别。算法在这些候选级别内寻找一致方案。若只有场景风险变化而对象影响未改变，可以给出专项复核记录，不能为了让优化有解而随意改级。'

# 09 / incidence rows + grade vectors, very concrete coupling example
p=Page('联合优化：满足约束，同时减少不必要的调整','08  为什么等级需要一起考虑',
       '示意前提：候选级别已复核，且级别调整会改变该角色未来可获取的字段。')
p.text(66,191,550,'四组已核验越线组合',26,C['ink'],True)
p.text(669,191,544,'比较满足同一组约束的方案',26,C['ink'],True)
for idx,members in enumerate([('A','B'),('A','C'),('A','D'),('B','C','D')]):
    y=255+idx*69
    p.text(66,y+5,52,f'{idx+1:02d}',19,C['muted'])
    for j,m in enumerate(members):
        p.label(121+j*81,y,65,m,C['blue'],C['bluep'],40,24)
    p.line(377,y+20,424,y+20,C['teal'],2,True)
    p.text(438,y+3,151,'目标 y',23,C['teal'],True)
p.text(67,558,532,'选择 A 与 D，可让这四组组合各缺少成员。',21,C['muted'])
p.rect(655,248,562,141,C['pale'],r=9)
p.text(680,265,400,'全部上调到 3 级',23,C['muted'],True)
p.text(680,314,400,'A 2→3   B 1→3   C 1→3   D 1→3',21,C['ink'])
p.text(1095,270,95,'7',51,C['muted'],True,'center')
p.text(1100,337,93,'级次',18,C['muted'],'',align='center')
p.rect(655,408,562,141,C['tealp'],r=9)
p.text(680,425,400,'只上调 A、D',23,C['teal'],True)
p.text(680,474,400,'A 2→3   B 保持   C 保持   D 1→3',21,C['ink'])
p.text(1095,430,95,'3',51,C['teal'],True,'center')
p.text(1100,497,93,'级次',18,C['teal'],align='center')
p.takeaway('关键是“哪个调整能同时满足哪些组合约束”，而不是简单上调 M 最大的字段。')
p.cite('示意：目标 3 级，角色最多取得 2 级字段，各级次权重相同；已知背景保留；仅讨论这四组约束。')
p.notes='A、B、C、D 初始等级为 2、1、1、1。假设该角色未来只能取得 2 级及以下数据，目标是 3 级；四组已核验越线组合为 AB、AC、AD、BCD。全部上调到 3 级共 7 级次，而只上调 A、D 共 3 级次，也能覆盖这四条约束。该示意用于解释联合优化，并不表示必须这样给真实数据定级。成立前提是候选级别有影响依据、等级映射有效、字段尚可调整，固定背景没有单独暴露目标。未评估组合仍需检查。'

# 10 / actual GRPO mechanism, not merely four boxes
p=Page('GRPO：在规则硬约束内，逐步形成等级建议','09  联合等级决策的重点研究方法',
       '同一状态采样多条调整轨迹，在组内比较收益，学习更合适的调整顺序。')
p.label(65,184,1150,'硬约束贯穿全程：规则基线 · 合法候选级别 · 已知背景 / 披露要求 · 审批边界',C['blue'],C['bluep'],47,22)
p.node(64,269,282,178,'状态','当前等级、组合风险\n剩余约束、调整代价',C['blue'],title_size=28)
p.node(407,269,284,178,'GRPO 策略','采样多条调整轨迹\n组内相对优势更新',C['teal'],C['tealp'],28)
p.node(753,269,209,178,'动作','选字段调一级\n或申请结束',C['blue'],title_size=28)
p.node(1021,269,195,178,'环境反馈','更新可得范围\n重算剩余约束',C['blue'],title_size=25)
for a,b in ((346,407),(691,753),(962,1021)):p.line(a+8,356,b-9,356,C['muted'],2,True)
p.line(1118,453,1118,480,C['blue'],2)
p.line(1118,480,203,480,C['blue'],2)
p.line(203,480,203,451,C['blue'],2,True)
p.text(409,483,620,'调整一个字段 → 其余组合的可得性随之变化',21,C['blue'],align='center')
p.rect(64,542,720,66,C['pale'],r=7)
p.text(82,559,680,'奖励：已核验越线减少 − 调整代价',26,C['ink'],True)
p.rect(807,542,409,66,C['tealp'],r=7)
p.text(826,553,369,'通过一致性检查与补充验证\n才形成可提交的建议',20,C['teal'],True,lh=1.35)
p.cite('算法保留 KL 约束等训练机制；不允许用奖励抵消规则违规。整数规划与覆盖贪心作为对照，结果见备查 A2。')
p.notes='GRPO 的重点是组内比较：从同一状态生成多条合法调整轨迹，用各轨迹回报相对组内水平的优势更新策略，并保留 KL 约束等稳定机制。状态包含当前等级、已核验风险、候选级别、剩余组合约束和调整代价；动作是合法的一级上调或申请结束。环境根据已确认的映射更新未来可得范围，历史已知 B 保留。奖励鼓励减少越线并控制调整代价，规则通过动作可行域限制，不作为可被业务收益抵消的罚项。顺序求解不自动证明强化学习必要，现有小规模实验应与整数规划及覆盖贪心比较。'

# 11 / validation and optimization are coupled, stop condition exact
p=Page('验证闭环：建议形成后，再检查剩余可得组合','10  让等级建议有足够证据支撑',
       '扫描用于提高效率，补充验证用于发现漏项；一次性验证不能代替闭环。')
p.node(65,220,482,128,'01  通用模型扫描','估计组合能力，提出重点核验对象',C['teal'],C['tealp'],26)
p.node(734,220,482,128,'02  形成等级建议','在当前证据与硬约束下运行 GRPO',C['blue'],title_size=26)
p.node(734,420,482,132,'03  补充验证','核验剩余可得组合与未确认约束',C['teal'],C['tealp'],26)
p.node(65,420,482,132,'04  更新证据并复算','发现遗漏或误报，修正约束与建议',C['blue'],title_size=26)
p.line(561,284,720,284,C['muted'],2,True)
p.line(975,355,975,407,C['muted'],2,True)
p.line(720,486,561,486,C['muted'],2,True)
p.line(304,406,304,358,C['muted'],2,True)
p.text(577,356,127,'迭代至\n范围内完成验证',20,C['muted'],align='center',lh=1.55)
p.takeaway('结论必须连同范围提交：数据版本、模型、目标、背景与已核验组合规模。',C['teal'])
p.cite('依据：项目 E2/E2b 验证记录。范围内无残余，不代表所有模型、未来数据与更大组合都安全。')
p.notes='已有实验发现，估计值偏低的组合可能根本不会被选入一次性验证清单，优化结果仍留有残余。因此形成等级建议后，需要核验剩余可得范围中的组合，并验证作为约束的可疑集合以纠正误报，再返回求解。结束条件应包含声明范围内的核验完成与约束检查通过。E2b 使用已有重训结果表模拟验证调用，零残余仅相对于这张固定表。真实项目还需要独立样本、时间顺序留出和模型范围核验。'

# 12 / deliverable and lifecycle
p=Page('输出可审批的等级清单，随变化持续复评','11  交付与落地',
       '让每一项等级建议都能追溯到规则、场景、推断证据与审批记录。')
p.rect(64,197,515,362,C['pale'],r=9)
p.text(88,217,463,'一项完整的分级记录',27,C['ink'],True)
rows=[('对象与分类','字段 / 数据集 / 产品及业务路径'),('规则基线','规则等级、依据条款与版本'),('补充证据','目标、背景、能力、验证范围'),('等级建议','建议级别、复核理由与约束'),('最终记录','审批结论、目录版本、复评条件')]
for i,(a,b) in enumerate(rows):
    y=275+i*52
    p.text(88,y,145,a,20,C['muted'])
    p.text(235,y,319,b,19,C['ink'],True)
p.text(630,211,585,'审批之后，才固化正式结果',27,C['blue'],True)
for x,w,lab in [(630,158,'业务复核'),(837,170,'审批确认'),(1056,158,'目录固化')]:
    p.label(x,279,w,lab,C['blue'],C['bluep'],64,23)
for x in (797,1016):p.line(x,311,x+31,311,C['blue'],2,True)
p.text(630,396,575,'触发复评的变化',25,C['ink'],True)
for i,(lab,col) in enumerate([('数据与产品变化',C['blue']),('背景与主体变化',C['teal']),('规则与时效变化',C['blue'])]):
    y=450+i*42
    p.circle(632,y+8,8,col)
    p.text(654,y,548,lab,23,C['muted'])
p.takeaway('规则有依据，风险有证据，建议可解释，结果可审批，变化可复评。')
p.cite('落地前确认：有效规则版本、目标及阈值、真实授权与历史背景、试点数据；本次不预填真实南网等级。')
p.notes='最终交付不是一个模型分数，而是一项完整分级记录：对象与业务分类、规则基线、推断证据、建议级别、复核理由以及审批与复评信息。业务部门复核后进入审批，批准结果固化到目录。数据加工、字段汇聚、主体和已知背景变化，以及规则修订或时效变化，触发下一轮复评。第一步先选择一个业务场景，确认目标、阈值和实际权限，接入历史数据完成闭环试点。'

# A1 / formal definitions, written in plain scalable math text
p=Page('备查：风险量与等级一致性条件','A1  定义与前提',
       '以下是研究评估量与约束的形式化表达，不是南网法规中的数值分级规则。',appendix=True)
p.label(64,183,184,'风险量的定义',C['teal'],C['tealp'])
p.text(65,235,1147,'Δᵢ(T) = V(B ∪ T ∪ {i}) − V(B ∪ T)',30,C['ink'],True)
p.text(65,294,1147,'M⁽⁰⁾ = Δᵢ(∅)     M⁽ᴷ⁾ = max Δᵢ(T)     Γ⁽ᴷ⁾ = M⁽ᴷ⁾ − M⁽⁰⁾',28,C['teal'],True)
p.text(66,341,1147,'最大值在可行背景 T 中取：来自候选背景池，不含 B 或字段 i，字段数不超过 K。',20,C['muted'])
p.line(64,394,1216,394,C['line'],1)
p.label(64,420,184,'一致性检查')
p.text(66,472,1147,'对该主体仍受保护的目标：所有纳入检查的可得组合 S 均满足 V(B ∪ S) ≤ τ。',25,C['ink'],True)
p.text(66,525,1128,'目标：在允许的候选级别内，减少加权调整代价；约束：规则基线、披露要求与一致性。',23,C['blue'])
p.text(66,578,1140,'B 已足以暴露、成员不可调整或映射不成立时，报告冲突并人工复核，不承诺一定有解。',22,C['amber'])
p.cite('V 是固定经验协议下的样本外能力；K 是评估预算；τ 按目标与业务确认；以上省略目标和场景下标。')
p.notes='风险量都以同一目标、同一场景和同一评估协议为前提。M 的实测最大值通常是已搜索背景族内的最大值，不能代表全部未知背景。检查一致性使用最终集合能力 V，而不是直接用 M 作等级阈值。等级与可见范围的对应关系必须业务确认；实际非嵌套角色要分别列出可得集合。GRPO 学习的是可行候选空间中的联合调整，不能通过改级使历史已知信息消失。'

# A2 / truthful evidence: solver results and validation are separate
p=Page('备查：已有验证结果与下一步','A2  研究证据与适用范围',
       '重点展示 GRPO 的研究机制，同时保留精确基线和已有实验的真实结论。',appendix=True)
p.text(64,187,565,'GRPO 与基线的代价比较',25,C['ink'],True)
p.text(677,187,533,'验证闭环的作用',25,C['ink'],True)
p.text(64,230,565,'相对精确最优的平均代价（越接近 1 越好）',18,C['muted'])
cols=[64,323,463]
for x,w,lab in [(64,240,'方法'),(323,122,'见过的组'),(463,142,'未见过的组')]:
    p.text(x,279,w,lab,20,C['muted'],True)
for j,(name,a,b,col) in enumerate([('精确最优','1.000','1.000',C['muted']),('覆盖贪心 + 回退','1.021','1.016',C['muted']),('GRPO + 回退','1.010','1.008',C['teal'])]):
    y=326+j*53
    if j==2:p.rect(58,y-4,559,47,C['tealp'],r=5)
    for x,w,lab in [(64,253,name),(323,122,a),(463,142,b)]:p.text(x,y,w,lab,21,col,j==2)
p.text(66,512,549,'当前规模精确求解平均约 8 ms。\nGRPO 相对贪心小幅改善，扩展价值仍待验证。',20,C['muted'],lh=1.6)
p.line(641,190,641,588,C['line'],1)
p.text(678,251,510,'使用估计值先验后，需要核验的集合比例',21,C['muted'])
p.text(681,296,192,'83%',52,C['muted'],True)
p.line(860,335,943,335,C['teal'],3,True)
p.text(977,296,220,'39%',52,C['teal'],True)
p.text(679,369,530,'不用先验                         使用先验',19,C['muted'])
p.text(679,430,526,'闭环在既有真值表内达到零残余；\n验证调用通过查表模拟，非新增重训。',21,C['ink'],lh=1.6)
p.text(679,520,529,'下一步：真实权限与等级、南网样本、\n更大组合范围与动态场景。',21,C['blue'],lh=1.55)
p.cite('来源：01 号实验 WORKLOG、e3_summary.csv、e2b_summary.csv；公开电力数据、随机等级/代价、≤35 字段、≤3 字段组合、2 种子。')
p.notes='本页根据 2026-10-07 实验记录更新。E3 使用两个种子的留出实例结果，GRPO 加回退相对最优的平均代价为 1.010 和 1.008，覆盖贪心为 1.021 和 1.016。E1 中整数规划平均约 8 毫秒，是当前小规模任务的重要精确基线。E2b 使用估计先验时平均核验 38.71% 的集合，不用先验时为 82.84%；所谓核验是在已有专用模型结果表中查询，不代表新增实际训练。随机等级与代价、有限字段和组合规模都应随结论一起报告，不能当作南网现场效果。'

def write_svg(page):
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
           f'<title id="title">{escape(page.title)}</title><desc id="desc">{escape(page.notes)}</desc>',
           '<rect width="1280" height="720" fill="#FFFFFF"/>']
    colors=set(i.get('color') for i in page.items if i['kind']=='line' and i.get('arrow'))
    parts.append('<defs>')
    for c in colors:parts.append(f'<marker id="a{c}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#{c}"/></marker>')
    parts.append('</defs>')
    for o in page.items:
        k=o['kind']
        if k in ('rect','circle'):
            stroke=f' stroke="#{o["stroke"]}" stroke-width="{o["sw"]}"' if o.get('stroke') else ''
            if k=='rect':parts.append(f'<rect x="{o["x"]}" y="{o["y"]}" width="{o["w"]}" height="{o["h"]}" rx="{o["r"]}" fill="#{o["fill"]}"{stroke}/>')
            else:parts.append(f'<ellipse cx="{o["x"]+o["w"]/2}" cy="{o["y"]+o["h"]/2}" rx="{o["w"]/2}" ry="{o["h"]/2}" fill="#{o["fill"]}"{stroke}/>')
        elif k=='line':
            extras=(' stroke-dasharray="6 5"' if o.get('dash') else '')+(f' marker-end="url(#a{o["color"]})"' if o.get('arrow') else '')
            parts.append(f'<line x1="{o["x"]}" y1="{o["y"]}" x2="{o["x2"]}" y2="{o["y2"]}" stroke="#{o["color"]}" stroke-width="{o["width"]}"{extras}/>')
        else:
            anchor={'left':'start','center':'middle','right':'end'}[o['align']]
            x=o['x']+({'left':0,'center':o['w']/2,'right':o['w']}[o['align']])
            weight='700' if o['bold'] else '400'
            parts.append(f'<text font-family="{FONT}, Microsoft YaHei, sans-serif" font-size="{o["size"]}" font-weight="{weight}" fill="#{o["color"]}" text-anchor="{anchor}">')
            for j,line in enumerate(o['text'].split('\n')):
                parts.append(f'<tspan x="{x}" y="{o["y"]+o["size"]*1.02+j*o["lineHeight"]}">{escape(line)}</tspan>')
            parts.append('</text>')
    parts.append('</svg>')
    return '\n'.join(parts)

def main():
    for name in ('svg','src','notes','work','preview'):(ROOT/name).mkdir(exist_ok=True)
    issues=[]
    for s in slides:
        for i,o in enumerate(s.items):
            if o['kind']=='line':
                if not(0<=min(o['x'],o['x2']) and max(o['x'],o['x2'])<=W and 0<=min(o['y'],o['y2']) and max(o['y'],o['y2'])<=H):issues.append([s.n,i,'line bounds'])
            elif o['x']<0 or o['y']<0 or o['x']+o['w']>W or o['y']+o['h']>H:issues.append([s.n,i,'bounds',o.get('text','')])
        (ROOT/'svg'/f'{s.n:02d}.svg').write_text(write_svg(s))
    data=dict(width=W,height=H,font=FONT,title='面向推断风险的数据分类分级体系',slides=[dict(number=s.n,title=s.title,section=s.section,notes=s.notes,appendix=s.appendix,items=s.items) for s in slides])
    (ROOT/'src/scenes.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    (ROOT/'notes/03_逐页讲稿.md').write_text('# 逐页讲稿\n\n'+'\n\n'.join(f'## {s.n:02d} {s.title.replace(chr(10),"")}\n\n{s.notes}' for s in slides)+'\n')
    (ROOT/'work/scene_bounds.json').write_text(json.dumps({'slides':len(slides),'out_of_bounds':issues},ensure_ascii=False,indent=2))
    print('Slides:',len(slides),'Elements:',sum(len(s.items) for s in slides),'Bounds:',issues)
    assert not issues
    cards='\n'.join(f'<article><a href="svg/{s.n:02d}.svg"><img src="svg/{s.n:02d}.svg" alt="{escape(s.title)}" loading="lazy"/></a><p>{s.n:02d} · {escape(s.title.replace(chr(10),""))}</p></article>' for s in slides)
    gallery='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>南网分类分级体系 · 逐页预览</title><style>body{margin:40px;font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;background:#f4f7fa;color:#172b3a}h1{font-size:28px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:28px}article{margin:0}img{width:100%;box-shadow:0 2px 12px #172b3a12;background:white}p{font-size:16px}a{color:inherit}@media(max-width:560px){body{margin:16px}main{grid-template-columns:1fr}}@media print{article{break-after:page}main{display:block}body{margin:0}p,h1,header{display:none}}</style><header><h1>面向推断风险的数据分类分级体系</h1><p>12 页主讲 + 2 页备查 · 点击任意页面打开独立 SVG · 中文字体：Noto Sans CJK SC（回退 Microsoft YaHei）</p></header><main>'+cards+'</main></html>'
    (ROOT/'逐页预览.html').write_text(gallery)

if __name__=='__main__':main()
