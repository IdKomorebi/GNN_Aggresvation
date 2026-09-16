import os; os.environ.setdefault("LOSS","g"); os.environ.setdefault("WD","0"); os.environ.setdefault("STEPS","5000")
import numpy as np, torch, sys
src=open(__file__.replace('interp.py','pilot_geometry2.py')).read().split("LEVELS = [0.05")[0]
exec(src)
CONF=sys.argv[1]
items=by_conf[CONF]
fit=[x for x in items if x[1] in ("single","pair","triple_rand")]
fs=[x[0] for x in fit]; fv=np.array([x[2] for x in fit])
# 复制 fit_geom 但保留参数
torch.manual_seed(0); d=8
X=torch.tensor(X_of(fs),dtype=torch.float64,device=dev); y=torch.tensor(np.clip(fv,0,1),dtype=torch.float64,device=dev)
U=(0.3*torch.randn(n,d,dtype=torch.float64,device=dev)).requires_grad_(); ar=torch.randn(d,dtype=torch.float64,device=dev).requires_grad_(); s=torch.tensor(1.0,dtype=torch.float64,device=dev,requires_grad=True)
opt=torch.optim.Adam([U,ar,s],lr=0.03); gl=lambda x:-0.5*torch.log(1-torch.clamp(x,0,1-1e-3))
def pred(Xb):
    a=ar/ar.norm()*torch.sigmoid(s*3); M=torch.einsum("bi,ide->bde",Xb,torch.einsum("id,ie->ide",U,U))+torch.eye(d,dtype=torch.float64,device=dev)
    return a@a-torch.linalg.solve(M,a.expand(len(Xb),d).unsqueeze(-1)).squeeze(-1)@a
for t in range(5000):
    opt.zero_grad(); p=pred(X); loss=((gl(p)-gl(y))**2).mean()+((p-y)**2).mean(); loss.backward(); opt.step()
a=(ar/ar.norm()*torch.sigmoid(s*3)).detach().cpu().numpy(); Un=U.detach().cpu().numpy()
ah=a/np.linalg.norm(a); sig=Un@ah; nui=Un-np.outer(sig,ah); nn_=np.linalg.norm(nui,axis=1)
print(f"{CONF}: ||a||²(可推断上限)={a@a:.3f}")
order=np.argsort(-np.abs(sig)/np.sqrt(1+np.linalg.norm(Un,axis=1)**2))
print("字段 | 信号分量|s| | 干扰范数 | 单独v | 信噪比 s²/(1+|u|²)")
vs=np.array([ (sig[i]**2)*(a@a)/(1+np.linalg.norm(Un[i])**2) for i in range(n)])
for i in np.argsort(-nn_)[:8]:
    print(f"  {fields[i]:40} {abs(sig[i]):6.2f} {nn_[i]:7.2f} {vs[i]:6.3f}")
# 最强协同对：真值 v_ij - max(v_i, v_j)，与干扰方向余弦
truth={ids:v for ids,g_,v in items}
syn=sorted(((truth[(i,j)]-max(truth[(i,)],truth[(j,)]),i,j) for i in range(n) for j in range(i+1,n)),reverse=True)[:6]
cos=lambda i,j: nui[i]@nui[j]/(nn_[i]*nn_[j]+1e-9)
print("真值最强协同对 | syn | 干扰方向余弦 | 信号比 s_i/|nu_i| vs s_j/|nu_j|")
for sy,i,j in syn: print(f"  {fields[i]} + {fields[j]}: syn={sy:.3f} cos={cos(i,j):+.3f}  {sig[i]/nn_[i]:+.3f} vs {sig[j]/nn_[j]:+.3f}")
allc=[abs(cos(i,j)) for i in range(n) for j in range(i+1,n)]
print(f"  全部字段对 |cos| 中位数={np.median(allc):.3f}")
