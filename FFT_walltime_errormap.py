import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve
import time, os, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = '/mnt/user-data/outputs'
os.makedirs(OUT, exist_ok=True)
matplotlib.rc('font', **{'family': 'DejaVu Sans', 'size': 13})
print(f"Device: {DEVICE}")

def save_fig(fig, name):
    # plt.show() version -- displays inline, saves nothing
    plt.show()

# =====================================================================
#  HELMHOLTZ PIECES  
# =====================================================================
K = 1.0; F_AMP = 2.0*np.pi**2 - K**2
def h_u_exact(x, y):  return np.sin(np.pi*x)*np.sin(np.pi*y)
def h_f_source(x, y): return F_AMP*np.sin(np.pi*x)*np.sin(np.pi*y)

def helmholtz_fdm(N=256):
    h = 1.0/N; ni = N-1
    dm = np.full(ni, 2.0); do_ = np.full(ni-1, -1.0)
    T = diags([do_, dm, do_], [-1,0,1], shape=(ni,ni), format='csr')
    Is = eye(ni, format='csr')
    L = (kron(T,Is)+kron(Is,T))/h**2 - K**2*eye(ni*ni, format='csr')
    xi = np.linspace(h,1-h,ni); yi = np.linspace(h,1-h,ni)
    Xm,Ym = np.meshgrid(xi,yi)
    ui = spsolve(L, h_f_source(Xm.flatten(),Ym.flatten())).reshape(ni,ni)
    xg = np.linspace(0,1,N+1); yg = np.linspace(0,1,N+1)
    Uf = np.zeros((N+1,N+1)); Uf[1:N,1:N] = ui
    Xf,Yf = np.meshgrid(xg,yg)
    return xg, yg, h_u_exact(Xf,Yf), Uf

class LSSAAct(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((n,), 0.5))
        self.beta  = nn.Parameter(torch.ones(n))
        self.gamma = nn.Parameter(torch.full((n,), float(np.pi)))
    def forward(self, z):
        a = torch.clamp(self.alpha,0.01,0.99).unsqueeze(0)
        b = self.beta.unsqueeze(0); g = self.gamma.unsqueeze(0)
        return a*torch.tanh(b*z)+(1-a)*torch.sin(g*z)*torch.exp(-0.5*z**2)

class HelmholtzLSSA(nn.Module):
    def __init__(self, layers=None):
        super().__init__()
        if layers is None: layers=[2,128,128,128,128,128,1]
        self.depth=len(layers)-1
        lins,acts=[],[]
        for i in range(self.depth-1):
            l=nn.Linear(layers[i],layers[i+1])
            nn.init.xavier_uniform_(l.weight,gain=1.0); nn.init.zeros_(l.bias)
            lins.append(l); acts.append(LSSAAct(layers[i+1]))
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
    ux=torch.autograd.grad(u,xc,torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    uxx=torch.autograd.grad(ux,xc,torch.ones_like(ux),create_graph=True,retain_graph=True)[0]
    uy=torch.autograd.grad(u,yc,torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    uyy=torch.autograd.grad(uy,yc,torch.ones_like(uy),create_graph=True,retain_graph=True)[0]
    res=-(uxx+uyy)-K**2*u-fc
    return torch.mean(res**2)+w_bc*torch.mean((model(xb,yb)-ub)**2)

def helmholtz_sample():
    Nc=10000; Nb=200
    px=np.random.permutation(Nc); py=np.random.permutation(Nc)
    xc=(px+np.random.rand(Nc))/Nc; yc=(py+np.random.rand(Nc))/Nc
    fc=h_f_source(xc,yc)
    s=np.linspace(0,1,Nb)
    xb=np.concatenate([s,s,np.zeros(Nb),np.ones(Nb)])
    yb=np.concatenate([np.zeros(Nb),np.ones(Nb),s,s]); ub=np.zeros(len(xb))
    def T(a): return torch.tensor(a,dtype=torch.float32,device=DEVICE).reshape(-1,1)
    return T(xc),T(yc),T(fc),T(xb),T(yb),T(ub)

def train_helmholtz_lssa():
    torch.manual_seed(SEED); np.random.seed(SEED)
    model=HelmholtzLSSA().to(DEVICE)
    xc,yc,fc,xb,yb,ub=helmholtz_sample()
    opt=torch.optim.Adam(model.parameters(),lr=5e-4,weight_decay=1e-6)
    sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=4000,T_mult=2,eta_min=1e-5)
    print("[Helmholtz LSSA] Adam ...")
    for ep in range(1,20001):
        opt.zero_grad(); loss=helmholtz_loss(model,xc,yc,fc,xb,yb,ub)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sch.step(ep)
        if ep%5000==0: print(f"  Ep {ep} | {loss.item():.3e}")
    for rnd in range(4):
        olb=torch.optim.LBFGS(model.parameters(),lr=1.0,max_iter=500,max_eval=2000,
            history_size=50,tolerance_grad=1e-10,tolerance_change=1e-12,line_search_fn='strong_wolfe')
        def cl():
            olb.zero_grad(); l=helmholtz_loss(model,xc,yc,fc,xb,yb,ub); l.backward(); return l
        olb.step(cl)
    return model

def helmholtz_eval(model,xg,yg,batch=4096):
    model.eval(); Xm,Ym=np.meshgrid(xg,yg); xf=Xm.flatten(); yf=Ym.flatten()
    n=len(xf); uf=np.zeros(n)
    with torch.no_grad():
        for s in range(0,n,batch):
            e=min(s+batch,n)
            xt=torch.tensor(xf[s:e],dtype=torch.float32,device=DEVICE).reshape(-1,1)
            yt=torch.tensor(yf[s:e],dtype=torch.float32,device=DEVICE).reshape(-1,1)
            uf[s:e]=model(xt,yt).cpu().numpy().flatten()
    return uf.reshape(len(yg),len(xg))

# =====================================================================
#  ALLEN-CAHN PIECES  
# =====================================================================
EPS2=0.0001
def ac_fdm(Nx=512,Nt=5000,eps2=EPS2):
    x=np.linspace(-1,1,Nx+1); dx=x[1]-x[0]; dt=1./Nt; r=eps2*dt/dx**2
    N=Nx; u=x[:N]**2*np.cos(np.pi*x[:N])
    d=np.full(N,1+2*r); od=np.full(N-1,-r)
    A=diags([od,d,od],[-1,0,1],shape=(N,N),format='lil')
    A[0,N-1]=-r; A[N-1,0]=-r; A=A.tocsr()
    se=max(1,Nt//200); steps=list(range(0,Nt+1,se))
    U=np.zeros((len(steps),Nx+1)); ptr=0; U[ptr]=np.append(u,u[0]); ptr+=1
    for n in range(Nt):
        rhs=u+dt*(u-u**3); u=spsolve(A,rhs)
        if (n+1) in steps and ptr<len(steps): U[ptr]=np.append(u,u[0]); ptr+=1
    return x, np.array(steps)/Nt, U

class ACLSSA(nn.Module):
    def __init__(self, layers=None):
        super().__init__()
        if layers is None: layers=[2,128,128,128,128,128,1]
        self.depth=len(layers)-1
        lins,acts=[],[]
        for i in range(self.depth-1):
            l=nn.Linear(layers[i],layers[i+1])
            nn.init.xavier_uniform_(l.weight,gain=1.0); nn.init.zeros_(l.bias)
            lins.append(l); acts.append(LSSAAct(layers[i+1]))
        out=nn.Linear(layers[-2],layers[-1])
        nn.init.xavier_uniform_(out.weight,gain=1.0); nn.init.zeros_(out.bias)
        lins.append(out)
        self.linears=nn.ModuleList(lins); self.activations=nn.ModuleList(acts)
    def forward(self,x,t):
        z=torch.cat([x,t],dim=1)
        for i in range(self.depth-1): z=self.activations[i](self.linears[i](z))
        return self.linears[-1](z)

class ACStd(nn.Module):
    def __init__(self, activation, layers=None):
        super().__init__()
        if layers is None: layers=[2,128,128,128,128,128,1]
        self.depth=len(layers)-1
        self.af = torch.tanh if activation=='tanh' else F.silu  # silu = swish
        lins=[]
        for i in range(self.depth-1):
            l=nn.Linear(layers[i],layers[i+1])
            nn.init.xavier_uniform_(l.weight,gain=1.0); nn.init.zeros_(l.bias); lins.append(l)
        out=nn.Linear(layers[-2],layers[-1])
        nn.init.xavier_uniform_(out.weight,gain=1.0); nn.init.zeros_(out.bias); lins.append(out)
        self.linears=nn.ModuleList(lins)
    def forward(self,x,t):
        z=torch.cat([x,t],dim=1)
        for i in range(self.depth-1): z=self.af(self.linears[i](z))
        return self.linears[-1](z)

def ac_loss(model,xc,tc,xb,tb,xi,ti,ui,eps2=EPS2):
    xc=xc.requires_grad_(True); tc=tc.requires_grad_(True)
    u=model(xc,tc)
    ux=torch.autograd.grad(u,xc,torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    uxx=torch.autograd.grad(ux,xc,torch.ones_like(ux),create_graph=True,retain_graph=True)[0]
    ut=torch.autograd.grad(u,tc,torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    res=ut-eps2*uxx-u+u**3
    lp=torch.mean(res**2)
    lb=torch.mean((model(xb,tb)-model(-xb,tb))**2)
    li=torch.mean((model(xi,ti)-ui)**2)
    return 1.0*lp+10.0*lb+20.0*li

def ac_sample():
    Nc=10000; Ni=3000; Nb=200; Nic=300
    px=np.random.permutation(Nc); pt=np.random.permutation(Nc)
    xc=(px+np.random.rand(Nc))/Nc*2-1; tc=(pt+np.random.rand(Nc))/Nc
    xii=np.concatenate([np.random.uniform(-1,-0.6,Ni//2),np.random.uniform(0.6,1,Ni//2)])
    tii=np.random.uniform(0.3,1,Ni)
    xc=np.concatenate([xc,xii]); tc=np.concatenate([tc,tii])
    xb=np.full(Nb,-1.0); tb=np.random.uniform(0,1,Nb)
    xi=np.random.uniform(-1,1,Nic); ti=np.zeros(Nic); ui=xi**2*np.cos(np.pi*xi)
    def T(a): return torch.tensor(a,dtype=torch.float32,device=DEVICE).reshape(-1,1)
    return T(xc),T(tc),T(xb),T(tb),T(xi),T(ti),T(ui)

def train_ac(model, capture_loss=False):
    """Trains an Allen-Cahn model. If capture_loss, records (wallclock, loss)
    pairs with real timestamps across Adam and L-BFGS phases."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    model=model.to(DEVICE)
    xc,tc,xb,tb,xi,ti,ui=ac_sample()
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-6)
    sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=5000,T_mult=2,eta_min=1e-5)
    times=[]; losses=[]; t0=time.time()
    for ep in range(1,30001):
        opt.zero_grad(); loss=ac_loss(model,xc,tc,xb,tb,xi,ti,ui)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sch.step(ep)
        if capture_loss and (ep%100==0):
            times.append(time.time()-t0); losses.append(loss.item())
        if ep%15000==0: print(f"    Ep {ep} | {loss.item():.3e}")
    for rnd in range(4):
        olb=torch.optim.LBFGS(model.parameters(),lr=1.0,max_iter=500,max_eval=2000,
            history_size=50,tolerance_grad=1e-10,tolerance_change=1e-12,line_search_fn='strong_wolfe')
        rec=[]
        def cl():
            olb.zero_grad(); l=ac_loss(model,xc,tc,xb,tb,xi,ti,ui); l.backward()
            rec.append(l.item()); return l
        olb.step(cl)
        if capture_loss:
            # append L-BFGS points with current wallclock
            for j,lv in enumerate(rec):
                times.append(time.time()-t0); losses.append(lv)
    return model, np.array(times), np.array(losses)

def ac_eval(model,x_fdm,t_fdm):
    model.eval(); U=np.zeros((len(t_fdm),len(x_fdm)))
    with torch.no_grad():
        for i,tv in enumerate(t_fdm):
            xt=torch.tensor(x_fdm,dtype=torch.float32,device=DEVICE).reshape(-1,1)
            tt=torch.full_like(xt,float(tv))
            U[i]=model(xt,tt).cpu().numpy().flatten()
    return U

# =====================================================================
#  RUN 
# =====================================================================
print("\n########## Helmholtz LSSA (FFT) ##########")
xg,yg,Uex_h,Ufdm_h = helmholtz_fdm()
m_h = train_helmholtz_lssa()
U_h = helmholtz_eval(m_h,xg,yg)
print("  saved U_pred_lssa_helmholtz.npy")

print("\n########## Allen-Cahn LSSA / Swish / tanh ##########")
x_ac,t_ac,Uref_ac = ac_fdm()
print("  training LSSA (Allen-Cahn) [loss-vs-time captured] ...")
m_lssa_ac, t_lssa, l_lssa = train_ac(ACLSSA(), capture_loss=True)
U_lssa_ac = ac_eval(m_lssa_ac,x_ac,t_ac)
print("  training Swish (Allen-Cahn) ...")
m_sw_ac, _, _ = train_ac(ACStd('swish'), capture_loss=False)
U_sw_ac = ac_eval(m_sw_ac,x_ac,t_ac)
print("  training tanh (Allen-Cahn) [loss-vs-time captured] ...")
m_tanh_ac, t_tanh, l_tanh = train_ac(ACStd('tanh'), capture_loss=True)

# ──PLOT: FFT SPECTRUM ────────────────────────────────────────
print("\n FFT of Helmholtz output ...")
Ny,Nx = U_h.shape
dx = (xg[1]-xg[0])
U = U_h - U_h.mean()
Fm = np.fft.fftshift(np.abs(np.fft.fft2(U)))
fx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
iy,ix = np.unravel_index(np.argmax(Fm), Fm.shape)
peak_fx = abs(fx[ix])
print(f"  dominant peak |fx| = {peak_fx:.3f} cyc/unit (single sharp mode)")
spec = Fm[iy,:]/Fm[iy,:].max()
fig,ax=plt.subplots(figsize=(8,5.5)); fig.patch.set_facecolor('white'); ax.set_facecolor('white')
ax.plot(fx,spec,color='#C0392B',lw=2.0,label='LSSA output spectrum')
ax.axvline(peak_fx,color='#1E8449',lw=2.0,ls='--',label=f'Dominant peak $|f_x|={peak_fx:.2f}$ cyc/unit')
ax.axvline(-peak_fx,color='#1E8449',lw=2.0,ls='--')
ax.set_xlabel('Spatial frequency $f_x$ (cycles/unit)',fontsize=13)
ax.set_ylabel('Normalized magnitude',fontsize=13)
ax.set_title('Spatial Fourier Spectrum of LSSA-PINN Output\n(2D Helmholtz)',fontweight='bold',fontsize=14,pad=10)
ax.set_xlim(-2.5,2.5); ax.legend(fontsize=11,framealpha=0.9)
ax.grid(True,linestyle='--',linewidth=0.6,alpha=0.45); ax.tick_params(labelsize=11)
plt.tight_layout(pad=1.0); save_fig(fig,'helmholtz_output_fft_spectrum.png')

# ──PLOT: LOSS vs WALL-CLOCK ──────────────────────────────────
print("Loss vs wall-clock ...")
fig,ax=plt.subplots(figsize=(8.5,5.5)); fig.patch.set_facecolor('white'); ax.set_facecolor('white')
ax.semilogy(t_lssa,l_lssa,color='#C0392B',lw=1.8,label='LSSA')
ax.semilogy(t_tanh,l_tanh,color='#1A5276',lw=1.8,label='tanh')
ax.set_xlabel('Wall-clock time (s)',fontsize=13); ax.set_ylabel('Training loss',fontsize=13)
ax.set_title('Loss vs Wall-Clock Time\n1D Allen--Cahn',fontweight='bold',fontsize=14,pad=10)
ax.legend(fontsize=12,framealpha=0.9)
ax.grid(True,which='both',linestyle='--',linewidth=0.5,alpha=0.45); ax.tick_params(labelsize=11)
plt.tight_layout(pad=1.0); save_fig(fig,'allencahn_loss_vs_walltime.png')

# ──PLOT: SQUARED-ERROR SPATIAL MAP LSSA vs SWISH ────────────
print("Squared-error maps LSSA vs Swish ...")
X,T_=np.meshgrid(x_ac,t_ac)
err_lssa=(U_lssa_ac-Uref_ac)**2
err_sw=(U_sw_ac-Uref_ac)**2
vmax=max(err_lssa.max(),err_sw.max())
fig,axes=plt.subplots(1,2,figsize=(13,5.2)); fig.patch.set_facecolor('white')
for ax,(E,title) in zip(axes,[(err_lssa,'LSSA squared error'),(err_sw,'Swish squared error')]):
    c=ax.contourf(X,T_,E,levels=120,cmap='hot_r',vmin=0,vmax=vmax)
    cb=fig.colorbar(c,ax=ax,shrink=0.9,pad=0.02,format='%.1e'); cb.ax.tick_params(labelsize=10)
    ax.set_title(title,fontweight='bold',fontsize=13,pad=8)
    ax.set_xlabel('$x$',fontsize=13); ax.set_ylabel('$t$',fontsize=13); ax.tick_params(labelsize=11)
plt.tight_layout(pad=1.2); save_fig(fig,'allencahn_squared_error_lssa_vs_swish.png')

print("\nDONE. Three figures + saved arrays in", OUT)
print(f"  LSSA total Adam+LBFGS wall time (Allen-Cahn): {t_lssa[-1]:.0f}s")
print(f"  tanh total Adam+LBFGS wall time (Allen-Cahn): {t_tanh[-1]:.0f}s")
