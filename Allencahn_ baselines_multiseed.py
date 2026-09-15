"""
=============================================================================
MULTI-SEED BASELINE COMPARISON — 1D Allen-Cahn Equation
Reviewer 1 request: multi-seed statistics for ALL baselines (not just LSSA)
=============================================================================
PURPOSE:
  Reviewer noted the reproducibility study ran only LSSA across seeds while
  baselines were single-seed, making the Allen-Cahn comparison asymmetric.
  This script runs the four fixed-activation baselines (tanh, GELU, Swish,
  SELU) across the SAME four seeds used for LSSA (42, 123, 2024, 7),
  producing a symmetric mean +/- std comparison.

  Total runs: 4 activations x 4 seeds = 16 trainings.

EVERYTHING IS IDENTICAL to the finalized LSSA Allen-Cahn comparison:
  PDE:   u_t = eps2*u_xx + u - u^3,  x in [-1,1], t in [0,1]
  IC:    u(x,0) = x^2*cos(pi*x)
  BC:    u(-1,t) = u(1,t)  [periodic]
  eps2:  0.0001
  FDM:   IMEX, Nx=512, Nt=5000
  Net:   5 x 128 MLP; SELU uses LeCun-normal init, others Xavier-uniform
  Adam:  30000 epochs, CosineAnnealingWarmRestarts T0=5000, T_mult=2
  LBFGS: 4 rounds x 500 iters
  Loss:  w_pde=1, w_bc=10, w_ic=20
  Smpl:  N_col=10000 + N_interface=3000, N_bc=200, N_ic=300
  Seeds: 42, 123, 2024, 7  (matched to LSSA reproducibility study)

  LSSA reference (already established, from Section 5.6):
    L2: mean 0.157% +/- 0.139% | L1: mean 0.074% +/- 0.045%

OUTPUT: text-only — per-activation per-seed L2/L1/loss, plus
        per-activation mean +/- std across the four seeds, and a final
        5-method comparison block (LSSA reference + 4 baselines).
=============================================================================
"""

import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import time, warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS2 = 0.0001
SEEDS = [42, 123, 2024, 7]
ACTIVATIONS = ['tanh', 'gelu', 'swish', 'selu']

print(f"Device: {DEVICE}")
print(f"Seeds : {SEEDS}")
print(f"Activations: {ACTIVATIONS}")
print(f"Total runs: {len(ACTIVATIONS)} x {len(SEEDS)} = {len(ACTIVATIONS)*len(SEEDS)}")
print("="*70)

# ── FDM REFERENCE (computed once — seed-independent) ─────────────────
def compute_fdm(Nx=512, Nt=5000, eps2=EPS2):
    print("[FDM] Computing IMEX Allen-Cahn reference ...")
    t0 = time.time()
    x = np.linspace(-1, 1, Nx+1); dx = x[1]-x[0]; dt = 1./Nt; r = eps2*dt/dx**2
    N = Nx; u = x[:N]**2 * np.cos(np.pi*x[:N])
    d = np.full(N, 1+2*r); od = np.full(N-1, -r)
    A = diags([od, d, od], [-1, 0, 1], shape=(N, N), format='lil')
    A[0, N-1] = -r; A[N-1, 0] = -r; A = A.tocsr()
    store_every = max(1, Nt//200); steps = list(range(0, Nt+1, store_every))
    U_stored = np.zeros((len(steps), Nx+1)); ptr = 0
    U_stored[ptr] = np.append(u, u[0]); ptr += 1
    for n in range(Nt):
        rhs = u + dt*(u - u**3); u = spsolve(A, rhs)
        if (n+1) in steps and ptr < len(steps):
            U_stored[ptr] = np.append(u, u[0]); ptr += 1
    t_stored = np.array(steps)/Nt
    print(f"      done ({time.time()-t0:.1f}s)")
    return x, t_stored, U_stored

# ── STANDARD PINN (fixed activation) ─────────────────────────────────
class StandardPINN(nn.Module):
    def __init__(self, activation='tanh', layers=None):
        super().__init__()
        if layers is None: layers = [2, 128, 128, 128, 128, 128, 1]
        self.depth = len(layers) - 1
        self.af_name = activation
        if activation == 'tanh':   self.af = torch.tanh
        elif activation == 'gelu': self.af = F.gelu
        elif activation == 'swish':self.af = F.silu
        elif activation == 'selu': self.af = F.selu
        else: raise ValueError(f"Unknown: {activation}")
        lins = []
        for i in range(self.depth-1):
            l = nn.Linear(layers[i], layers[i+1])
            if activation == 'selu':
                nn.init.kaiming_normal_(l.weight, mode='fan_in', nonlinearity='linear')
            else:
                nn.init.xavier_uniform_(l.weight, gain=1.0)
            nn.init.zeros_(l.bias); lins.append(l)
        out = nn.Linear(layers[-2], layers[-1])
        nn.init.xavier_uniform_(out.weight, gain=1.0); nn.init.zeros_(out.bias)
        lins.append(out); self.linears = nn.ModuleList(lins)
    def forward(self, x, t):
        z = torch.cat([x, t], dim=1)
        for i in range(self.depth-1): z = self.af(self.linears[i](z))
        return self.linears[-1](z)

def allencahn_loss(model, xc, tc, xb, tb, xi, ti, ui, eps2=EPS2):
    xc = xc.requires_grad_(True); tc = tc.requires_grad_(True)
    u = model(xc, tc)
    ux = torch.autograd.grad(u, xc, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    uxx = torch.autograd.grad(ux, xc, torch.ones_like(ux), create_graph=True, retain_graph=True)[0]
    ut = torch.autograd.grad(u, tc, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    res = ut - eps2*uxx - u + u**3
    lp = torch.mean(res**2)
    lb = torch.mean((model(xb, tb) - model(-xb, tb))**2)
    li = torch.mean((model(xi, ti) - ui)**2)
    return 1.0*lp + 10.0*lb + 20.0*li

def sample_points(seed):
    rng = np.random.RandomState(seed)
    N_col = 10000; N_interface = 3000; N_bc = 200; N_ic = 300
    px = rng.permutation(N_col); pt = rng.permutation(N_col)
    xc = (px + rng.rand(N_col))/N_col*2 - 1
    tc = (pt + rng.rand(N_col))/N_col
    xi_int = np.concatenate([rng.uniform(-1., -0.6, N_interface//2),
                             rng.uniform(0.6, 1., N_interface//2)])
    ti_int = rng.uniform(0.3, 1., N_interface)
    xc = np.concatenate([xc, xi_int]); tc = np.concatenate([tc, ti_int])
    xb = np.full(N_bc, -1.0); tb = rng.uniform(0, 1, N_bc)
    xi = rng.uniform(-1, 1, N_ic); ti = np.zeros(N_ic)
    ui = xi**2 * np.cos(np.pi*xi)
    def T(a): return torch.tensor(a, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    return T(xc), T(tc), T(xb), T(tb), T(xi), T(ti), T(ui)

def train_one(activation, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = StandardPINN(activation=activation); model.to(DEVICE)
    xc, tc, xb, tb, xi, ti, ui = sample_points(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5000, T_mult=2, eta_min=1e-5)
    t0 = time.time()
    for ep in range(1, 30001):
        opt.zero_grad()
        loss = allencahn_loss(model, xc, tc, xb, tb, xi, ti, ui)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(ep)
        if ep % 15000 == 0:
            print(f"      [{activation} seed {seed}] Ep {ep} | Loss {loss.item():.3e}")
    for rnd in range(4):
        opt_lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=500,
            max_eval=2000, history_size=50, tolerance_grad=1e-10,
            tolerance_change=1e-12, line_search_fn='strong_wolfe')
        last_loss = [None]
        def closure():
            opt_lb.zero_grad()
            loss = allencahn_loss(model, xc, tc, xb, tb, xi, ti, ui)
            loss.backward(); last_loss[0] = loss.item(); return loss
        opt_lb.step(closure)
    total_t = time.time() - t0
    return model, last_loss[0], total_t

def evaluate(model, x_fdm, t_fdm):
    model.eval(); U = np.zeros((len(t_fdm), len(x_fdm)))
    with torch.no_grad():
        for i, ti in enumerate(t_fdm):
            xt = torch.tensor(x_fdm, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
            tt = torch.full_like(xt, float(ti))
            U[i] = model(xt, tt).cpu().numpy().flatten()
    return U

def compute_metrics(pred, ref):
    d = pred.flatten() - ref.flatten(); r = ref.flatten()
    L2 = np.linalg.norm(d)/(np.linalg.norm(r) + 1e-12)
    L1 = np.sum(np.abs(d))/(np.sum(np.abs(r)) + 1e-12)
    return L2, L1

# ── RUN 4 ACTIVATIONS x 4 SEEDS ──────────────────────────────────────
x_fdm, t_fdm, U_ref = compute_fdm()

all_results = {af: [] for af in ACTIVATIONS}

for af in ACTIVATIONS:
    print("\n" + "="*70)
    print(f"  ACTIVATION: {af.upper()}")
    print("="*70)
    for seed in SEEDS:
        print(f"  [{af} | seed {seed}] training ...")
        model, final_loss, total_t = train_one(af, seed)
        U_pred = evaluate(model, x_fdm, t_fdm)
        L2, L1 = compute_metrics(U_pred, U_ref)
        all_results[af].append(dict(seed=seed, L2=L2, L1=L1, loss=final_loss, time=total_t))
        print(f"  [{af} | seed {seed}] L2={L2*100:.4f}% | L1={L1*100:.4f}% | "
              f"loss={final_loss:.3e} | {total_t:.0f}s")

# ── PER-ACTIVATION SUMMARY ───────────────────────────────────────────
print("\n" + "="*70)
print("  PER-SEED RESULTS — 1D Allen-Cahn — 4 baselines x 4 seeds")
print("="*70)
for af in ACTIVATIONS:
    print(f"\n  {af.upper()}:")
    print(f"    {'Seed':<8}{'L2 (%)':>12}{'L1 (%)':>12}{'Loss':>14}")
    for r in all_results[af]:
        print(f"    {r['seed']:<8}{r['L2']*100:>12.4f}{r['L1']*100:>12.4f}{r['loss']:>14.3e}")
    L2v = np.array([r['L2'] for r in all_results[af]])*100
    L1v = np.array([r['L1'] for r in all_results[af]])*100
    print(f"    {'Mean':<8}{L2v.mean():>12.4f}{L1v.mean():>12.4f}")
    print(f"    {'Std':<8}{L2v.std(ddof=1):>12.4f}{L1v.std(ddof=1):>12.4f}")

# ── FINAL 5-METHOD COMPARISON TABLE (mean +/- std) ───────────────────
print("\n" + "="*70)
print("  FINAL COMPARISON — mean +/- std across 4 seeds — 1D Allen-Cahn")
print("="*70)
print(f"  {'Method':<12}{'L2 mean+/-std (%)':>24}{'L1 mean+/-std (%)':>24}")
print("-"*70)
print(f"  {'LSSA':<12}{'0.157 +/- 0.139':>24}{'0.074 +/- 0.045':>24}  [ref, Sec 5.6]")
for af in ACTIVATIONS:
    L2v = np.array([r['L2'] for r in all_results[af]])*100
    L1v = np.array([r['L1'] for r in all_results[af]])*100
    l2s = f"{L2v.mean():.3f} +/- {L2v.std(ddof=1):.3f}"
    l1s = f"{L1v.mean():.3f} +/- {L1v.std(ddof=1):.3f}"
    print(f"  {af.upper():<12}{l2s:>24}{l1s:>24}")
print("="*70)

# ── LaTeX-READY ROWS ─────────────────────────────────────────────────
print("\nLaTeX table rows (mean +/- std):")
print("-"*70)
print(f"  LSSA (proposed) & 0.157 $\\pm$ 0.139 & 0.074 $\\pm$ 0.045 \\\\")
print("  \\midrule")
for af in ACTIVATIONS:
    L2v = np.array([r['L2'] for r in all_results[af]])*100
    L1v = np.array([r['L1'] for r in all_results[af]])*100
    pretty = {'tanh':'tanh','gelu':'GELU','swish':'Swish','selu':'SELU'}[af]
    print(f"  {pretty:<16} & {L2v.mean():.3f} $\\pm$ {L2v.std(ddof=1):.3f} & "
          f"{L1v.mean():.3f} $\\pm$ {L1v.std(ddof=1):.3f} \\\\")
