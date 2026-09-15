"""
Plot generator for LSSA Helmholtz finalized results.
Conditions applied:
  - Each plot is an individual PNG, 600 dpi
  - No accuracy numbers anywhere on any plot
  - All original training code unchanged
  - Individual frames:
      (1) Heatmaps: Exact | LSSA | FDM | Error
      (2) Cross-sections at y=0.25, 0.50, 0.75
      (3) Training loss history
      (4) Activation shape (tanh / init / learned)
      (5) alpha histogram
      (6) gamma histogram
"""

import torch, torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve
import time, os, warnings
warnings.filterwarnings('ignore')
# ── CONFIG ────────────────────────────────────────────────────────────────────
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K=1.0; F_AMP=2.0*np.pi**2-K**2
OUT='/mnt/user-data/outputs'
os.makedirs(OUT, exist_ok=True)

FONT={'family':'DejaVu Sans','size':13}
matplotlib.rc('font',**FONT)

def save(fig, name):
    p=os.path.join(OUT, name)
    fig.savefig(p, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Saved: {p}")
    plt.close(fig)

# ── EXACT & SOURCE ────────────────────────────────────────────────────────────
def u_exact(x,y): return np.sin(np.pi*x)*np.sin(np.pi*y)
def f_source(x,y): return F_AMP*np.sin(np.pi*x)*np.sin(np.pi*y)

# ── FDM ───────────────────────────────────────────────────────────────────────
def compute_fdm(N=256):
    print(f"[FDM] Helmholtz N={N} ...")
    t0=time.time(); h=1.0/N; ni=N-1
    dm=np.full(ni,2.0); do_=np.full(ni-1,-1.0)
    T=diags([do_,dm,do_],[-1,0,1],shape=(ni,ni),format='csr')
    Is=eye(ni,format='csr')
    L=(kron(T,Is)+kron(Is,T))/h**2-K**2*eye(ni*ni,format='csr')
    xi=np.linspace(h,1-h,ni); yi=np.linspace(h,1-h,ni)
    Xm,Ym=np.meshgrid(xi,yi)
    ui=spsolve(L,f_source(Xm.flatten(),Ym.flatten())).reshape(ni,ni)
    xg=np.linspace(0,1,N+1); yg=np.linspace(0,1,N+1)
    Uf=np.zeros((N+1,N+1)); Uf[1:N,1:N]=ui
    Xf,Yf=np.meshgrid(xg,yg); Uex=u_exact(Xf,Yf)
    print(f"      done ({time.time()-t0:.1f}s)")
    return xg, yg, Uex, Uf

# ── LSSA MODEL ────────────────────────────────────────────────────────────────
class LSSAActivation(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.alpha=nn.Parameter(torch.full((n,),0.5))
        self.beta =nn.Parameter(torch.ones(n))
        self.gamma=nn.Parameter(torch.full((n,),float(np.pi)))
    def forward(self,z):
        a=torch.clamp(self.alpha,0.01,0.99).unsqueeze(0)
        b=self.beta.unsqueeze(0); g=self.gamma.unsqueeze(0)
        return a*torch.tanh(b*z)+(1.0-a)*torch.sin(g*z)*torch.exp(-0.5*z**2)

class LSSA_PINN_Helmholtz(nn.Module):
    def __init__(self,layers=None):
        super().__init__()
        if layers is None: layers=[2,128,128,128,128,128,1]
        self.depth=len(layers)-1
        lins,acts=[],[]
        for i in range(self.depth-1):
            l=nn.Linear(layers[i],layers[i+1])
            nn.init.xavier_uniform_(l.weight,gain=1.0); nn.init.zeros_(l.bias)
            lins.append(l); acts.append(LSSAActivation(layers[i+1]))
        out=nn.Linear(layers[-2],layers[-1])
        nn.init.xavier_uniform_(out.weight,gain=1.0); nn.init.zeros_(out.bias)
        lins.append(out)
        self.linears=nn.ModuleList(lins); self.activations=nn.ModuleList(acts)
    def forward(self,x,y):
        z=torch.cat([2.0*x-1.0,2.0*y-1.0],dim=1)
        for i in range(self.depth-1): z=self.activations[i](self.linears[i](z))
        return self.linears[-1](z)

def helmholtz_loss(model,xc,yc,fc,xb,yb,ub,w_bc=200.0):
    xc=xc.requires_grad_(True); yc=yc.requires_grad_(True)
    u=model(xc,yc)
    ux =torch.autograd.grad(u, xc,torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    uxx=torch.autograd.grad(ux,xc,torch.ones_like(ux),create_graph=True,retain_graph=True)[0]
    uy =torch.autograd.grad(u, yc,torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    uyy=torch.autograd.grad(uy,yc,torch.ones_like(uy),create_graph=True,retain_graph=True)[0]
    res=-(uxx+uyy)-K**2*u-fc
    lp=torch.mean(res**2); lb=torch.mean((model(xb,yb)-ub)**2)
    return lp+w_bc*lb, lp.item(), lb.item()

def sample_points(N_col=10000,N_bc_edge=200):
    px=np.random.permutation(N_col); py=np.random.permutation(N_col)
    xc=(px+np.random.rand(N_col))/N_col
    yc=(py+np.random.rand(N_col))/N_col
    fc=f_source(xc,yc)
    s=np.linspace(0,1,N_bc_edge)
    xb=np.concatenate([s,s,np.zeros(N_bc_edge),np.ones(N_bc_edge)])
    yb=np.concatenate([np.zeros(N_bc_edge),np.ones(N_bc_edge),s,s])
    ub=np.zeros(len(xb))
    def T(a): return torch.tensor(a,dtype=torch.float32,device=DEVICE).reshape(-1,1)
    return T(xc),T(yc),T(fc),T(xb),T(yb),T(ub)

def train(model):
    model.to(DEVICE)
    xc,yc,fc,xb,yb,ub=sample_points()
    opt=torch.optim.Adam(model.parameters(),lr=5e-4,weight_decay=1e-6)
    sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=4000,T_mult=2,eta_min=1e-5)
    h_loss,h_pde,h_bc=[],[],[]; conv_ep=20000; t0=time.time()
    for ep in range(1,20001):
        opt.zero_grad()
        loss,lp,lb=helmholtz_loss(model,xc,yc,fc,xb,yb,ub)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sch.step(ep)
        v=loss.item(); h_loss.append(v); h_pde.append(lp); h_bc.append(lb)
        if conv_ep==20000 and lp<1.0: conv_ep=ep
        if ep%4000==0: print(f"  Ep {ep} | Loss {v:.3e}")
    adam_t=time.time()-t0; print(f"Adam: {adam_t:.0f}s")
    all_lb=[]; t_lb=time.time()
    for rnd in range(4):
        rh=[]
        opt_lb=torch.optim.LBFGS(model.parameters(),lr=1.0,max_iter=500,max_eval=2000,
            history_size=50,tolerance_grad=1e-10,tolerance_change=1e-12,line_search_fn='strong_wolfe')
        def closure():
            opt_lb.zero_grad()
            loss,_,_=helmholtz_loss(model,xc,yc,fc,xb,yb,ub)
            loss.backward(); rh.append(loss.item()); return loss
        opt_lb.step(closure); all_lb.extend(rh)
        print(f"  LBFGS round {rnd+1}: {len(rh)} calls | loss={rh[-1]:.4e}")
    return dict(h_loss=h_loss,h_pde=h_pde,h_bc=h_bc,lbfgs_h=all_lb,
                conv_ep=conv_ep,adam_t=adam_t,lbfgs_t=time.time()-t_lb,total_t=time.time()-t0)

def evaluate(model,xg,yg,batch=4096):
    model.eval()
    Xm,Ym=np.meshgrid(xg,yg); xf=Xm.flatten(); yf=Ym.flatten()
    n=len(xf); uf=np.zeros(n)
    with torch.no_grad():
        for s in range(0,n,batch):
            e=min(s+batch,n)
            xt=torch.tensor(xf[s:e],dtype=torch.float32,device=DEVICE).reshape(-1,1)
            yt=torch.tensor(yf[s:e],dtype=torch.float32,device=DEVICE).reshape(-1,1)
            uf[s:e]=model(xt,yt).cpu().numpy().flatten()
    return uf.reshape(len(yg),len(xg))

# ── TRAIN ─────────────────────────────────────────────────────────────────────
print("Running FDM..."); xg,yg,Uex,Ufdm=compute_fdm()
print("Training LSSA-PINN...")
model=LSSA_PINN_Helmholtz(); info=train(model)
print("Evaluating..."); Upred=evaluate(model,xg,yg)

d=Upred.flatten()-Uex.flatten(); r=Uex.flatten()
met=dict(L2=np.linalg.norm(d)/(np.linalg.norm(r)+1e-12),
         L1=np.sum(np.abs(d))/(np.sum(np.abs(r))+1e-12),
         Linf=np.max(np.abs(d)))
print(f"L2={met['L2']*100:.4f}% | Linf={met['Linf']:.4e}")

Xm,Ym=np.meshgrid(xg,yg)
err=np.abs(Upred-Uex); vmin,vmax=Uex.min(),Uex.max()

alphas,betas,gammas=[],[],[]
with torch.no_grad():
    for act in model.activations:
        alphas.extend(torch.clamp(act.alpha,0.01,0.99).cpu().numpy().tolist())
        betas.extend(act.beta.cpu().numpy().tolist())
        gammas.extend(act.gamma.cpu().numpy().tolist())

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1a — Exact solution only 
# ═════════════════════════════════════════════════════════════════════════════
fig,ax=plt.subplots(figsize=(6,5.5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
ce=ax.contourf(Xm,Ym,Uex,levels=100,cmap='RdBu_r',vmin=vmin,vmax=vmax)
cbe=fig.colorbar(ce,ax=ax,shrink=0.90,pad=0.02,format='%.3f')
cbe.ax.tick_params(labelsize=11)
ax.set_title('Exact Solution',fontweight='bold',fontsize=14,pad=10)
ax.set_xlabel('$x$',fontsize=13); ax.set_ylabel('$y$',fontsize=13)
ax.set_aspect('equal'); ax.tick_params(labelsize=11)
plt.tight_layout(pad=1.0)
plt.show()
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1b — Heatmaps: FDM | LSSA | Error  
# ═════════════════════════════════════════════════════════════════════════════
fig,axes=plt.subplots(1,3,figsize=(18,5.5))
fig.patch.set_facecolor('white')
for ax,(data,title) in zip(axes[:2],
    [(Ufdm,'FDM Reference'),
     (Upred,'LSSA-PINN Prediction')]):
    cp=ax.contourf(Xm,Ym,data,levels=100,cmap='RdBu_r',vmin=vmin,vmax=vmax)
    cbp=fig.colorbar(cp,ax=ax,shrink=0.90,pad=0.02,format='%.3f')
    cbp.ax.tick_params(labelsize=11)
    ax.set_title(title,fontweight='bold',fontsize=13,pad=8)
    ax.set_xlabel('$x$',fontsize=13); ax.set_ylabel('$y$',fontsize=13)
    ax.set_aspect('equal'); ax.tick_params(labelsize=11)
c4=axes[2].contourf(Xm,Ym,err,levels=100,cmap='hot_r')
cb4=fig.colorbar(c4,ax=axes[2],shrink=0.90,pad=0.02,format='%.2e')
cb4.ax.tick_params(labelsize=11)
axes[2].set_title('Pointwise Error',fontweight='bold',fontsize=13,pad=8)
axes[2].set_xlabel('$x$',fontsize=13); axes[2].set_ylabel('$y$',fontsize=13)
axes[2].set_aspect('equal'); axes[2].tick_params(labelsize=11)
plt.tight_layout(pad=1.2)
plt.show()
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Cross-sections at y=0.25, 0.50, 0.75
# ═════════════════════════════════════════════════════════════════════════════
fig,ax=plt.subplots(figsize=(8,5.5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
ya=np.array(yg)
colors_s=['#1A5276','#B7950B','#1E8449']
styles_ex=['--',':','-.']
for ys,col,ls in zip([0.25,0.50,0.75],colors_s,styles_ex):
    idx=np.argmin(np.abs(ya-ys))
    ax.plot(xg,Uex[idx],  ls, color=col,lw=2.2,label=f'Exact $y={ys}$')
    ax.plot(xg,Upred[idx],'-', color=col,lw=1.6,alpha=0.85,label=f'LSSA $y={ys}$')
ax.set_xlabel('$x$',fontsize=13); ax.set_ylabel('$u(x,y)$',fontsize=13)
ax.set_title('',
             fontweight='bold',fontsize=14,pad=10)
ax.legend(fontsize=11,ncol=2,framealpha=0.9)
ax.grid(True,linestyle='--',linewidth=0.6,alpha=0.5)
ax.tick_params(labelsize=11)
plt.tight_layout(pad=1.0)
plt.show()
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Training loss history
# ═════════════════════════════════════════════════════════════════════════════
fig,ax=plt.subplots(figsize=(9,5.5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
ea=np.arange(1,len(info['h_pde'])+1)
ax.semilogy(ea,info['h_pde'],color='#C0392B',lw=1.1,alpha=0.85,label='PDE residual $L_{pde}$')
ax.semilogy(ea,info['h_bc'], color='#2471A3',lw=1.1,alpha=0.85,label='BC residual $L_{bc}$')
tot=[info['h_pde'][i]+info['h_bc'][i] for i in range(len(ea))]
ax.semilogy(ea,tot,color='black',lw=2.0,label='Total (Adam)')
if info['lbfgs_h']:
    el=np.arange(len(ea)+1,len(ea)+len(info['lbfgs_h'])+1)
    ax.semilogy(el,info['lbfgs_h'],color='#7D3C98',lw=1.8,ls='--',
                label='L-BFGS (%d calls)' % len(info['lbfgs_h']))
if info['conv_ep']<len(ea):
    ax.axvline(info['conv_ep'],color='crimson',lw=1.5,ls=':',label='Convergence epoch')
ax.set_xlabel('Epoch',fontsize=13); ax.set_ylabel('Loss',fontsize=13)
ax.set_title('',fontweight='bold',fontsize=14,pad=10)
ax.legend(fontsize=11,framealpha=0.9)
ax.grid(True,which='both',linestyle='--',linewidth=0.5,alpha=0.45)
ax.tick_params(labelsize=11)
plt.tight_layout(pad=1.0)
plt.show()
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Activation shape
# ═════════════════════════════════════════════════════════════════════════════
fig,ax=plt.subplots(figsize=(7,5.5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
z=np.linspace(-4,4,800)
am=float(np.mean(alphas)); bm=float(np.mean(betas)); gm=float(np.mean(gammas))
phi_l=am*np.tanh(bm*z)+(1-am)*np.sin(gm*z)*np.exp(-0.5*z**2)
phi_0=0.5*np.tanh(z)+0.5*np.sin(np.pi*z)*np.exp(-0.5*z**2)
ax.plot(z,np.tanh(z),'--',color='#7F8C8D',lw=2.0,label='$\\tanh$')
ax.plot(z,phi_0,   ':',color='#2471A3',lw=2.0,label='LSSA (init)')
ax.plot(z,phi_l,   '-',color='#C0392B',lw=2.8,
        label=f'LSSA (learned)\n$\\bar\\alpha={am:.3f}$, $\\bar\\beta={bm:.3f}$, $\\bar\\gamma={gm:.4f}$')
ax.axhline(0,color='#AAAAAA',lw=0.8); ax.axvline(0,color='#AAAAAA',lw=0.8)
ax.set_xlabel('$z$',fontsize=13); ax.set_ylabel('$\\phi(z)$',fontsize=13)
ax.set_title('',fontweight='bold',fontsize=14,pad=10)
ax.legend(fontsize=11,framealpha=0.9)
ax.grid(True,linestyle='--',linewidth=0.6,alpha=0.45)
ax.tick_params(labelsize=11)
plt.tight_layout(pad=1.0)
plt.show()
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 5 — alpha histogram
# ═════════════════════════════════════════════════════════════════════════════
fig,ax=plt.subplots(figsize=(7,5.5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
ax.hist(alphas,bins=40,color='#1E8449',edgecolor='white',linewidth=0.6,alpha=0.90)
ax.axvline(float(np.mean(alphas)),color='#145A32',lw=2.5,ls='--',
           label=f'Mean $\\bar\\alpha = {float(np.mean(alphas)):.3f}$')
ax.axvline(0.5,color='#7F8C8D',lw=1.8,ls=':',label='Init $\\alpha_0 = 0.5$')
ax.set_xlabel('$\\alpha$',fontsize=13); ax.set_ylabel('Count',fontsize=13)
ax.set_title('',fontweight='bold',fontsize=14,pad=10)
ax.legend(fontsize=11,framealpha=0.9)
ax.grid(True,linestyle='--',linewidth=0.6,alpha=0.45)
ax.tick_params(labelsize=11)
plt.tight_layout(pad=1.0)
plt.show()
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 6 — gamma histogram  
# ═════════════════════════════════════════════════════════════════════════════
fig,ax=plt.subplots(figsize=(7,5.5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
ax.hist(gammas,bins=40,color='#2471A3',edgecolor='white',linewidth=0.6,alpha=0.90)
ax.axvline(float(np.mean(gammas)),color='#1A252F',lw=2.5,ls='--',
           label=f'Mean $\\bar\\gamma = {float(np.mean(gammas)):.4f}$')
ax.axvline(np.pi,color='#C0392B',lw=2.5,ls='-.',
           label=f'Solution frequency $\\pi = {np.pi:.4f}$')
ax.set_xlabel('$\\gamma$',fontsize=13); ax.set_ylabel('Count',fontsize=13)
ax.set_title('',
             fontweight='bold',fontsize=14,pad=10)
ax.legend(fontsize=11,framealpha=0.9)
ax.grid(True,linestyle='--',linewidth=0.6,alpha=0.45)
ax.tick_params(labelsize=11)
plt.tight_layout(pad=1.0)
plt.show()
plt.close(fig)

print(f"alpha={float(np.mean(alphas)):.4f} | beta={float(np.mean(betas)):.4f} | gamma={float(np.mean(gammas)):.4f}")
print(f"|gamma - pi| = {abs(float(np.mean(gammas))-np.pi):.4f}")
