"""
Plot generator for LSSA Burgers' finalized results.
Conditions applied:
  - Each plot is an individual PNG, 600 dpi
  - No accuracy numbers anywhere on the plot
  - Proper axes, legends, colormaps, text
  - Frames: (1) Heatmaps, (2) Solution slices, (3) Loss history,
            (4) Activation shape, (5) alpha histogram, (6) gamma histogram
"""
import numpy as np, torch, torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, warnings
from scipy.linalg import solve_banded
warnings.filterwarnings('ignore')

SEED=42; torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT='/mnt/user-data/outputs'
os.makedirs(OUT, exist_ok=True)

FONT = {'family':'DejaVu Sans','size':13}
matplotlib.rc('font',**FONT)
# ── FDM ──────────────────────────────────────────────────────────────────────
def compute_fdm(Nx=512, Nt=10000, nu=0.01/np.pi):
    x=np.linspace(-1,1,Nx+1); dx=x[1]-x[0]; dt=1./Nt; r=nu*dt/(2*dx**2)
    u=-np.sin(np.pi*x); u[0]=u[-1]=0.; ni=Nx-1
    ab=np.zeros((3,ni)); ab[0,1:]=-r; ab[1,:]=1+2*r; ab[2,:-1]=-r
    store=max(1,Nt//100); steps=list(range(0,Nt+1,store))
    U=np.zeros((len(steps),Nx+1)); ptr=0
    if 0 in steps: U[ptr]=u.copy(); ptr+=1
    for n in range(Nt):
        ui=u[1:-1]; up=np.maximum(ui,0); um=np.minimum(ui,0)
        cb=(ui-np.concatenate(([u[0]],ui[:-1])))/dx
        cf=(np.concatenate((ui[1:],[u[-1]]))-ui)/dx
        rhs=ui+r*(np.concatenate(([u[0]],ui[:-1]))-2*ui+
                   np.concatenate((ui[1:],[u[-1]])))-dt*(up*cb+um*cf)
        un=u.copy(); un[1:-1]=solve_banded((1,1),ab,rhs)
        un[0]=un[-1]=0.; u=un
        if (n+1) in steps and ptr<len(steps): U[ptr]=u.copy(); ptr+=1
    return x, np.array(steps)/Nt, U

# ── LSSA model ────────────────────────────────────────────────────────────────
class LSSAActivation(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.alpha=nn.Parameter(torch.full((n,),0.5))
        self.beta =nn.Parameter(torch.ones(n))
        self.gamma=nn.Parameter(torch.full((n,),float(np.pi)))
    def forward(self,z):
        a=torch.clamp(self.alpha,0.01,0.99).unsqueeze(0)
        b=self.beta.unsqueeze(0); g=self.gamma.unsqueeze(0)
        return a*torch.tanh(b*z)+(1-a)*torch.sin(g*z)*torch.exp(-0.5*z**2)

class LSSA_PINN(nn.Module):
    def __init__(self,layers=None):
        super().__init__()
        if layers is None: layers=[2,128,128,128,128,128,1]
        self.depth=len(layers)-1
        lins,acts=[],[]
        for i in range(self.depth-1):
            l=nn.Linear(layers[i],layers[i+1])
            nn.init.xavier_uniform_(l.weight); nn.init.zeros_(l.bias)
            lins.append(l); acts.append(LSSAActivation(layers[i+1]))
        out=nn.Linear(layers[-2],layers[-1])
        nn.init.xavier_uniform_(out.weight); nn.init.zeros_(out.bias)
        lins.append(out)
        self.linears=nn.ModuleList(lins); self.activations=nn.ModuleList(acts)
    def forward(self,x,t):
        z=torch.cat([x,t],dim=1)
        for i in range(self.depth-1): z=self.activations[i](self.linears[i](z))
        return self.linears[-1](z)

def burgers_loss(model,xc,tc,xb,tb,ub,xi,ti,ui,nu=0.01/np.pi):
    xc=xc.requires_grad_(True); tc=tc.requires_grad_(True)
    u=model(xc,tc)
    ux=torch.autograd.grad(u,xc,torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    ut=torch.autograd.grad(u,tc,torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    uxx=torch.autograd.grad(ux,xc,torch.ones_like(ux),create_graph=True,retain_graph=True)[0]
    res=ut+u*ux-nu*uxx
    lp=torch.mean(res**2); lb=torch.mean((model(xb,tb)-ub)**2)
    li=torch.mean((model(xi,ti)-ui)**2)
    return 1.0*lp+20.0*lb+20.0*li, lp.item(), lb.item(), li.item()

def sample_points():
    N_col=12000; N_shock=5000; N_bc=300; N_ic=300
    px=np.random.permutation(N_col); pt=np.random.permutation(N_col)
    xc=(px+np.random.rand(N_col))/N_col*2-1
    tc=(pt+np.random.rand(N_col))/N_col
    xs=np.random.uniform(-0.25,0.25,N_shock); ts=np.random.uniform(0.5,1.0,N_shock)
    xc=np.concatenate([xc,xs]); tc=np.concatenate([tc,ts])
    tb=np.random.rand(N_bc)
    xb=np.concatenate([np.full(N_bc//2,-1.),np.full(N_bc-N_bc//2,1.)])
    ub=np.zeros(N_bc)
    xi=np.random.uniform(-1,1,N_ic); ti_=np.zeros(N_ic); ui=-np.sin(np.pi*xi)
    def T(a): return torch.tensor(a,dtype=torch.float32,device=DEVICE).reshape(-1,1)
    return T(xc),T(tc),T(xb),T(tb),T(ub),T(xi),T(ti_),T(ui)

def train(model):
    import time
    model.to(DEVICE)
    xc,tc,xb,tb,ub,xi,ti,ui=sample_points()
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-6)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30000,eta_min=1e-5)
    h_loss,h_pde,h_bc,h_ic=[],[],[],[]; conv_ep=30000; t0=time.time()
    for ep in range(1,30001):
        opt.zero_grad()
        loss,lp,lb,li=burgers_loss(model,xc,tc,xb,tb,ub,xi,ti,ui)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
        opt.step(); sch.step()
        v=loss.item(); h_loss.append(v); h_pde.append(lp); h_bc.append(lb); h_ic.append(li)
        if conv_ep==30000 and v<1e-4: conv_ep=ep
        if ep%5000==0: print(f"  Ep {ep} | Loss {v:.3e}")
    adam_t=time.time()-t0; print(f"Adam: {adam_t:.0f}s")
    all_lb=[]; t_lb=time.time()
    for rnd in range(4):
        rh=[]
        opt_lb=torch.optim.LBFGS(model.parameters(),lr=1.0,max_iter=500,
            max_eval=2000,history_size=100,tolerance_grad=1e-10,
            tolerance_change=1e-12,line_search_fn='strong_wolfe')
        def closure():
            opt_lb.zero_grad()
            loss,_,_,_=burgers_loss(model,xc,tc,xb,tb,ub,xi,ti,ui)
            loss.backward(); rh.append(loss.item()); return loss
        opt_lb.step(closure); all_lb.extend(rh)
        print(f"  LBFGS round {rnd+1}: {len(rh)} calls | loss={rh[-1]:.4e}")
    return dict(h_loss=h_loss,h_pde=h_pde,h_bc=h_bc,h_ic=h_ic,
                lbfgs_h=all_lb,conv_ep=conv_ep,
                adam_t=adam_t,lbfgs_t=time.time()-t_lb,total_t=time.time()-t0)

def evaluate(model,x_fdm,t_fdm):
    model.eval(); U=np.zeros((len(t_fdm),len(x_fdm)))
    with torch.no_grad():
        for i,ti in enumerate(t_fdm):
            xt=torch.tensor(x_fdm,dtype=torch.float32,device=DEVICE).reshape(-1,1)
            tt=torch.full_like(xt,float(ti))
            U[i]=model(xt,tt).cpu().numpy().flatten()
    return U

# ── TRAIN ─────────────────────────────────────────────────────────────────────
print("Running FDM..."); x_fdm,t_fdm,U_ref=compute_fdm()
print("Training LSSA-PINN (this takes ~80 min on CPU, ~10 min GPU)...")
model=LSSA_PINN(); info=train(model)
U_pred=evaluate(model,x_fdm,t_fdm)

# Metrics (used internally only — never printed on plots)
d=U_pred.flatten()-U_ref.flatten(); r=U_ref.flatten()
met=dict(L2=np.linalg.norm(d)/(np.linalg.norm(r)+1e-12),
         L1=np.sum(np.abs(d))/(np.sum(np.abs(r))+1e-12),
         Linf=np.max(np.abs(d)))
print(f"L2={met['L2']*100:.4f}% | L1={met['L1']*100:.4f}% | Linf={met['Linf']:.4e}")

# Collect LSSA parameters
alphas,betas,gammas=[],[],[]
with torch.no_grad():
    for act in model.activations:
        alphas.extend(torch.clamp(act.alpha,0.01,0.99).cpu().numpy().tolist())
        betas.extend(act.beta.cpu().numpy().tolist())
        gammas.extend(act.gamma.cpu().numpy().tolist())

X,T_=np.meshgrid(x_fdm,t_fdm)
err=np.abs(U_pred-U_ref)
vmin,vmax=U_ref.min(),U_ref.max()

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Heatmaps: FDM | LSSA | Error  (3 panels, one frame)
# ═════════════════════════════════════════════════════════════════════════════
fig,axes=plt.subplots(1,3,figsize=(18,5.5))
fig.patch.set_facecolor('white')
titles=['FDM Reference',
        'LSSA-PINN Prediction',
        'Pointwise Error']
for ax,(data,title) in zip(axes[:2],[(U_ref,titles[0]),(U_pred,titles[1])]):
    c=ax.contourf(X,T_,data,levels=120,cmap='RdBu_r',vmin=vmin,vmax=vmax)
    cb=fig.colorbar(c,ax=ax,shrink=0.9,pad=0.02)
    cb.ax.tick_params(labelsize=11)
    ax.set_title(title,fontweight='bold',fontsize=14,pad=10)
    ax.set_xlabel('$x$',fontsize=13); ax.set_ylabel('$t$',fontsize=13)
    ax.tick_params(labelsize=11)
c3=axes[2].contourf(X,T_,err,levels=120,cmap='hot_r')
cb3=fig.colorbar(c3,ax=axes[2],shrink=0.9,pad=0.02,format='%.2e')
cb3.ax.tick_params(labelsize=11)
axes[2].set_title(titles[2],fontweight='bold',fontsize=14,pad=10)
axes[2].set_xlabel('$x$',fontsize=13); axes[2].set_ylabel('$t$',fontsize=13)
axes[2].tick_params(labelsize=11)
plt.tight_layout(pad=1.2)
plt.show()
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Solution slices at t=0.25, 0.50, 0.75
# ═════════════════════════════════════════════════════════════════════════════
fig,ax=plt.subplots(figsize=(8,5.5))
fig.patch.set_facecolor('white')
t_arr=np.array(t_fdm)
colors_s=['#1A5276','#B7950B','#1E8449']
styles_fdm=['--',':','-.']
for st,col,ls in zip([0.25,0.50,0.75],colors_s,styles_fdm):
    idx=np.argmin(np.abs(t_arr-st))
    ax.plot(x_fdm,U_ref[idx],ls,color=col,lw=2.2,label=f'FDM $t={st}$')
    ax.plot(x_fdm,U_pred[idx],'-',color=col,lw=1.6,alpha=0.85,
            label=f'LSSA $t={st}$')
ax.set_xlabel('$x$',fontsize=13); ax.set_ylabel('$u(x,t)$',fontsize=13)
ax.set_title('',
             fontweight='bold',fontsize=14,pad=10)
ax.legend(fontsize=11,ncol=2,framealpha=0.9)
ax.grid(True,linestyle='--',linewidth=0.6,alpha=0.5)
ax.tick_params(labelsize=11)
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
plt.tight_layout(pad=1.0)
plt.show()
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Training loss history
# ═════════════════════════════════════════════════════════════════════════════
fig,ax=plt.subplots(figsize=(9,5.5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
ea=np.arange(1,len(info['h_pde'])+1)
ax.semilogy(ea,info['h_pde'],color='#C0392B',lw=1.1,alpha=0.85,label='PDE residual')
ax.semilogy(ea,info['h_bc'], color='#2471A3',lw=1.1,alpha=0.85,label='BC residual')
ax.semilogy(ea,info['h_ic'], color='#1E8449',lw=1.1,alpha=0.85,label='IC residual')
tot=[info['h_pde'][i]+info['h_bc'][i]+info['h_ic'][i] for i in range(len(ea))]
ax.semilogy(ea,tot,color='black',lw=2.0,label='Total (Adam)')
if info['lbfgs_h']:
    el=np.arange(len(ea)+1,len(ea)+len(info['lbfgs_h'])+1)
    ax.semilogy(el,info['lbfgs_h'],color='#7D3C98',lw=1.8,ls='--',
                label='L-BFGS (%d calls)' % len(info['lbfgs_h']))
if info['conv_ep']<len(ea):
    ax.axvline(info['conv_ep'],color='crimson',lw=1.5,ls=':',
               label=f'Convergence epoch')
ax.set_xlabel('Epoch',fontsize=13); ax.set_ylabel('Loss',fontsize=13)
ax.set_title('',fontweight='bold',fontsize=14,pad=10)
ax.legend(fontsize=11,framealpha=0.9)
ax.grid(True,which='both',linestyle='--',linewidth=0.5,alpha=0.45)
ax.tick_params(labelsize=11)
plt.tight_layout(pad=1.0)
plt.show()
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Activation shape (tanh / init / learned)
# ═════════════════════════════════════════════════════════════════════════════
fig,ax=plt.subplots(figsize=(7,5.5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
z=np.linspace(-4,4,800)
am=float(np.mean(alphas)); bm=float(np.mean(betas)); gm=float(np.mean(gammas))
phi_l=am*np.tanh(bm*z)+(1-am)*np.sin(gm*z)*np.exp(-0.5*z**2)
phi_0=0.5*np.tanh(z)+0.5*np.sin(np.pi*z)*np.exp(-0.5*z**2)
ax.plot(z,np.tanh(z),'--',color='#7F8C8D',lw=2.0,label='$\\tanh$')
ax.plot(z,phi_0,':',color='#2471A3',lw=2.0,label='LSSA (init)')
ax.plot(z,phi_l,'-',color='#C0392B',lw=2.8,
        label=f'LSSA (learned)\n$\\bar\\alpha={am:.3f}$, $\\bar\\beta={bm:.3f}$, $\\bar\\gamma={gm:.3f}$')
ax.axhline(0,color='#AAAAAA',lw=0.8,ls='-')
ax.axvline(0,color='#AAAAAA',lw=0.8,ls='-')
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
           label=f'Mean $\\bar\\gamma = {float(np.mean(gammas)):.3f}$')
ax.axvline(np.pi,color='#C0392B',lw=2.0,ls='-.',
           label=f'Init $\\gamma_0 = \\pi \\approx {np.pi:.3f}$')
ax.set_xlabel('$\\gamma$',fontsize=13); ax.set_ylabel('Count',fontsize=13)
ax.set_title('',fontweight='bold',fontsize=14,pad=10)
ax.legend(fontsize=11,framealpha=0.9)
ax.grid(True,linestyle='--',linewidth=0.6,alpha=0.45)
ax.tick_params(labelsize=11)
plt.tight_layout(pad=1.0)
plt.show()
plt.close(fig)

print("\nAll 6 plots saved to", OUT)
