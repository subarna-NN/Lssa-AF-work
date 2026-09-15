"""
=============================================================================
GAMMA_0 INITIALIZATION SENSITIVITY SWEEP (EXTENSION) — 2D Helmholtz
Reviewer 1 (Comment 1/2) & Reviewer 2: prove gamma_0=pi is not merely
"starting at the answer" by showing behaviour across a range of gamma_0.
=============================================================================
PURPOSE:
  Extends the earlier 3-point study (gamma_0 = pi/3, pi, 3pi) with THREE
  new initializations to form a dense 6-point sensitivity sweep:

      NEW runs in THIS script:  gamma_0 = 0.5*pi, 2*pi, 5*pi
      ALREADY HAVE (do not rerun):
          gamma_0 = pi/3  -> gamma_bar=1.021, L2=0.0035%
          gamma_0 = pi    -> gamma_bar=2.972, L2=0.0027%
          gamma_0 = 3*pi  -> gamma_bar=7.057, L2=99.86%

  Merge the three new rows with the three existing rows to build the
  final 6-point table/curve.

EVERYTHING ELSE IS IDENTICAL to the finalized lssa_helmholtz_v2 and to
the earlier sensitivity script:
  PDE:   -(u_xx + u_yy) - k^2*u = f,  (x,y) in [0,1]^2
  BC:    u = 0 on all four edges
  Sol:   u* = sin(pi*x)*sin(pi*y),  k=1
  FDM:   5-point stencil, N=256
  Net:   5x128 MLP, Xavier init for weights
  Adam:  20000 epochs, CosineAnnealingWarmRestarts T0=4000, T_mult=2
  LBFGS: 4 rounds x 500 iters
  w_bc=200, N_col=10000, N_bc=800, seed=42
  alpha_0=0.5, beta_0=1.0 (unchanged); ONLY gamma_0 varies
=============================================================================
"""

import torch, torch.nn as nn
import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve
import time, warnings
warnings.filterwarnings('ignore')

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K = 1.0
F_AMP = 2.0 * np.pi**2 - K**2

print(f"Device: {DEVICE}")
print(f"Solution frequency: pi = {np.pi:.4f}")
print("="*70)

def u_exact(x, y):  return np.sin(np.pi*x) * np.sin(np.pi*y)
def f_source(x, y): return F_AMP * np.sin(np.pi*x) * np.sin(np.pi*y)

# ── FDM REFERENCE ────────────────────────────────────────────────────
def compute_fdm(N=256):
    print("[FDM] Computing Helmholtz reference ...")
    t0 = time.time(); h = 1.0/N; ni = N-1
    dm = np.full(ni, 2.0); do_ = np.full(ni-1, -1.0)
    T = diags([do_, dm, do_], [-1, 0, 1], shape=(ni, ni), format='csr')
    Is = eye(ni, format='csr')
    L = (kron(T, Is) + kron(Is, T)) / h**2 - K**2 * eye(ni*ni, format='csr')
    xi = np.linspace(h, 1-h, ni); yi = np.linspace(h, 1-h, ni)
    Xm, Ym = np.meshgrid(xi, yi)
    ui = spsolve(L, f_source(Xm.flatten(), Ym.flatten())).reshape(ni, ni)
    xg = np.linspace(0, 1, N+1); yg = np.linspace(0, 1, N+1)
    Uf = np.zeros((N+1, N+1)); Uf[1:N, 1:N] = ui
    Xf, Yf = np.meshgrid(xg, yg); Uex = u_exact(Xf, Yf)
    print(f"      done ({time.time()-t0:.1f}s)")
    return xg, yg, Uex, Uf

# ── LSSA with configurable gamma_0 ───────────────────────────────────
class LSSAActivation(nn.Module):
    def __init__(self, n, gamma_0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((n,), 0.5))
        self.beta  = nn.Parameter(torch.ones(n))
        self.gamma = nn.Parameter(torch.full((n,), float(gamma_0)))
    def forward(self, z):
        a = torch.clamp(self.alpha, 0.01, 0.99).unsqueeze(0)
        b = self.beta.unsqueeze(0); g = self.gamma.unsqueeze(0)
        return a*torch.tanh(b*z) + (1-a)*torch.sin(g*z)*torch.exp(-0.5*z**2)

class LSSA_PINN(nn.Module):
    def __init__(self, gamma_0, layers=None):
        super().__init__()
        if layers is None: layers = [2, 128, 128, 128, 128, 128, 1]
        self.depth = len(layers) - 1
        lins, acts = [], []
        for i in range(self.depth-1):
            l = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_uniform_(l.weight, gain=1.0); nn.init.zeros_(l.bias)
            lins.append(l); acts.append(LSSAActivation(layers[i+1], gamma_0))
        out = nn.Linear(layers[-2], layers[-1])
        nn.init.xavier_uniform_(out.weight, gain=1.0); nn.init.zeros_(out.bias)
        lins.append(out)
        self.linears = nn.ModuleList(lins); self.activations = nn.ModuleList(acts)
    def forward(self, x, y):
        z = torch.cat([2.0*x-1.0, 2.0*y-1.0], dim=1)
        for i in range(self.depth-1): z = self.activations[i](self.linears[i](z))
        return self.linears[-1](z)

def helmholtz_loss(model, xc, yc, fc, xb, yb, ub, w_bc=200.0):
    xc = xc.requires_grad_(True); yc = yc.requires_grad_(True)
    u = model(xc, yc)
    ux = torch.autograd.grad(u, xc, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    uxx = torch.autograd.grad(ux, xc, torch.ones_like(ux), create_graph=True, retain_graph=True)[0]
    uy = torch.autograd.grad(u, yc, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    uyy = torch.autograd.grad(uy, yc, torch.ones_like(uy), create_graph=True, retain_graph=True)[0]
    res = -(uxx + uyy) - K**2 * u - fc
    lp = torch.mean(res**2); lb = torch.mean((model(xb, yb) - ub)**2)
    return lp + w_bc*lb, lp.item(), lb.item()

def sample_points(N_col=10000, N_bc_edge=200):
    px = np.random.permutation(N_col); py = np.random.permutation(N_col)
    xc = (px + np.random.rand(N_col)) / N_col
    yc = (py + np.random.rand(N_col)) / N_col
    fc = f_source(xc, yc)
    s = np.linspace(0, 1, N_bc_edge)
    xb = np.concatenate([s, s, np.zeros(N_bc_edge), np.ones(N_bc_edge)])
    yb = np.concatenate([np.zeros(N_bc_edge), np.ones(N_bc_edge), s, s])
    ub = np.zeros(len(xb))
    def T(a): return torch.tensor(a, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    return T(xc), T(yc), T(fc), T(xb), T(yb), T(ub)

def train(model, label):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model.to(DEVICE)
    xc, yc, fc, xb, yb, ub = sample_points()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=4000, T_mult=2, eta_min=1e-5)
    t0 = time.time()
    print(f"[{label}] Adam training ...")
    for ep in range(1, 20001):
        opt.zero_grad()
        loss, lp, lb = helmholtz_loss(model, xc, yc, fc, xb, yb, ub)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(ep)
        if ep % 5000 == 0: print(f"  [{label}] Ep {ep} | Loss {loss.item():.3e}")
    for rnd in range(4):
        opt_lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=500,
            max_eval=2000, history_size=50, tolerance_grad=1e-10,
            tolerance_change=1e-12, line_search_fn='strong_wolfe')
        ll = [None]
        def closure():
            opt_lb.zero_grad()
            loss, _, _ = helmholtz_loss(model, xc, yc, fc, xb, yb, ub)
            loss.backward(); ll[0] = loss.item(); return loss
        opt_lb.step(closure)
    print(f"  [{label}] done {time.time()-t0:.0f}s | final loss={ll[0]:.4e}")

def evaluate(model, xg, yg, batch=4096):
    model.eval()
    Xm, Ym = np.meshgrid(xg, yg); xf = Xm.flatten(); yf = Ym.flatten()
    n = len(xf); uf = np.zeros(n)
    with torch.no_grad():
        for s in range(0, n, batch):
            e = min(s+batch, n)
            xt = torch.tensor(xf[s:e], dtype=torch.float32, device=DEVICE).reshape(-1, 1)
            yt = torch.tensor(yf[s:e], dtype=torch.float32, device=DEVICE).reshape(-1, 1)
            uf[s:e] = model(xt, yt).cpu().numpy().flatten()
    return uf.reshape(len(yg), len(xg))

def compute_metrics(pred, ref):
    d = pred.flatten() - ref.flatten(); r = ref.flatten()
    return (np.linalg.norm(d)/(np.linalg.norm(r)+1e-12),
            np.sum(np.abs(d))/(np.sum(np.abs(r))+1e-12))

def get_gamma_mean(model):
    gs = []
    with torch.no_grad():
        for act in model.activations:
            gs.extend(act.gamma.cpu().numpy().tolist())
    return float(np.mean(gs))

# ── NEW GAMMA_0 POINTS (three only) ──────────────────────────────────
new_points = [
    ('gamma0_0.5pi', 0.5*np.pi),
    ('gamma0_2pi',   2.0*np.pi),
    ('gamma0_5pi',   5.0*np.pi),
]

xg, yg, Uex, Ufdm = compute_fdm()

results = {}
for label, g0 in new_points:
    print("\n" + "="*70)
    print(f"  RUN: {label}  (gamma_0 = {g0:.4f} = {g0/np.pi:.2f}*pi)")
    print("="*70)
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = LSSA_PINN(gamma_0=g0)
    train(model, label)
    Upred = evaluate(model, xg, yg)
    L2, L1 = compute_metrics(Upred, Uex)
    gbar = get_gamma_mean(model)
    results[label] = dict(gamma0=g0, gamma_bar=gbar, L2=L2, L1=L1)
    print(f"  [{label}] gamma_0={g0:.4f} -> gamma_bar={gbar:.4f} | "
          f"L2={L2*100:.4f}% | L1={L1*100:.4f}% | |gbar-pi|={abs(gbar-np.pi):.4f}")

# ── MERGED 6-POINT TABLE (3 existing + 3 new) ────────────────────────
print("\n" + "="*70)
print("  GAMMA_0 SENSITIVITY SWEEP — full 6-point table")
print("  (existing 3 points hard-coded from earlier study for the table)")
print("="*70)

existing = {
    'pi/3': dict(gamma0=np.pi/3, gamma_bar=1.021, L2=0.000035, L1=0.000023),
    'pi':   dict(gamma0=np.pi,   gamma_bar=2.972, L2=0.000027, L1=0.000018),
    '3pi':  dict(gamma0=3*np.pi, gamma_bar=7.057, L2=0.9986,  L1=None),
}

print(f"  {'gamma_0':<12}{'gamma_0/pi':>12}{'gamma_bar':>12}{'|gbar-pi|':>12}{'L2 (%)':>12}")
print("-"*70)
# Assemble all six in ascending gamma_0 order
rows = []
rows.append((np.pi/3, existing['pi/3']['gamma_bar'], existing['pi/3']['L2']))
rows.append((0.5*np.pi, results['gamma0_0.5pi']['gamma_bar'], results['gamma0_0.5pi']['L2']))
rows.append((np.pi, existing['pi']['gamma_bar'], existing['pi']['L2']))
rows.append((2*np.pi, results['gamma0_2pi']['gamma_bar'], results['gamma0_2pi']['L2']))
rows.append((3*np.pi, existing['3pi']['gamma_bar'], existing['3pi']['L2']))
rows.append((5*np.pi, results['gamma0_5pi']['gamma_bar'], results['gamma0_5pi']['L2']))
for g0, gbar, L2 in rows:
    print(f"  {g0:<12.4f}{g0/np.pi:>12.2f}{gbar:>12.4f}{abs(gbar-np.pi):>12.4f}{L2*100:>12.4f}")
print("="*70)

print("\nLaTeX rows (fill existing pi/3, pi, 3pi from your earlier logs if more precise):")
for g0, gbar, L2 in rows:
    tag = f"{g0/np.pi:.2f}\\pi"
    print(f"  ${tag}$ & {gbar:.4f} & {abs(gbar-np.pi):.4f} & {L2*100:.4f} \\\\")
