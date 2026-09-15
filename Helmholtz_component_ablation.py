"""
=============================================================================
ABLATION STUDY — LSSA-PINN on 2D Helmholtz Equation
Reviewer 1 request: isolate the contribution of each LSSA parameter (α, β, γ)
=============================================================================
PURPOSE:
  Disable each LSSA parameter in turn and compare against Full LSSA:

    V1  Fixed γ         : α learnable, β learnable, γ frozen at π
    V2  Fixed α         : α frozen at 0.5, β learnable, γ learnable
    V3  Learnable β only: α frozen at 0.5, β learnable, γ frozen at π
    V4  Pure Gabor      : α frozen at 0.0 (kills tanh), β learnable, γ learnable

  Full LSSA reference (already established, single-seed=42):
    L2=0.0027%, L1=0.0018%, γ̄=2.972

EVERYTHING ELSE IS IDENTICAL to the finalized lssa_helmholtz_v2:
  PDE:   -(u_xx + u_yy) - k^2*u = f,  (x,y) in [0,1]^2
  BC:    u = 0 on all four edges
  Sol:   u* = sin(pi*x)*sin(pi*y),  k=1
  FDM:   5-point stencil, N=256
  Net:   5x128 MLP, Xavier init for weights
  Adam:  20000 epochs, CosineAnnealingWarmRestarts T0=4000, T_mult=2
  LBFGS: 4 rounds x 500 iters
  w_bc=200, N_col=10000, N_bc=800, seed=42
  alpha_0=0.5, beta_0=1.0, gamma_0=pi

OUTPUT: text-only results — per-variant L2, L1, final loss, trained mean
        (α, β, γ) — ready for direct insertion into a manuscript table.
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

# ── EXACT AND SOURCE ─────────────────────────────────────────────────
def u_exact(x, y):  return np.sin(np.pi*x) * np.sin(np.pi*y)
def f_source(x, y): return F_AMP * np.sin(np.pi*x) * np.sin(np.pi*y)

# ── FDM REFERENCE (identical to finalized version) ───────────────────
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

# ── ABLATION LSSA ACTIVATION ─────────────────────────────────────────
class AblationLSSAActivation(nn.Module):
    """
    LSSA with per-parameter freeze flags.
      alpha_init, beta_init, gamma_init  : starting values
      freeze_alpha/beta/gamma            : if True, that parameter is
                                           registered as a buffer (not
                                           trained) with the init value
                                           held constant throughout.
    """
    def __init__(self, n, alpha_init=0.5, beta_init=1.0, gamma_init=None,
                 freeze_alpha=False, freeze_beta=False, freeze_gamma=False):
        super().__init__()
        gamma_init = float(np.pi) if gamma_init is None else gamma_init

        if freeze_alpha:
            self.register_buffer('alpha', torch.full((n,), float(alpha_init)))
        else:
            self.alpha = nn.Parameter(torch.full((n,), float(alpha_init)))

        if freeze_beta:
            self.register_buffer('beta', torch.full((n,), float(beta_init)))
        else:
            self.beta = nn.Parameter(torch.full((n,), float(beta_init)))

        if freeze_gamma:
            self.register_buffer('gamma', torch.full((n,), float(gamma_init)))
        else:
            self.gamma = nn.Parameter(torch.full((n,), float(gamma_init)))

    def forward(self, z):
        a = torch.clamp(self.alpha, 0.0, 1.0).unsqueeze(0)
        b = self.beta.unsqueeze(0)
        g = self.gamma.unsqueeze(0)
        return a * torch.tanh(b * z) + (1.0 - a) * torch.sin(g * z) * torch.exp(-0.5 * z**2)


class AblationPINN(nn.Module):
    def __init__(self, variant_config, layers=None):
        super().__init__()
        if layers is None:
            layers = [2, 128, 128, 128, 128, 128, 1]
        self.depth = len(layers) - 1
        lins, acts = [], []
        for i in range(self.depth - 1):
            l = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_uniform_(l.weight, gain=1.0)
            nn.init.zeros_(l.bias)
            lins.append(l)
            acts.append(AblationLSSAActivation(layers[i+1], **variant_config))
        out = nn.Linear(layers[-2], layers[-1])
        nn.init.xavier_uniform_(out.weight, gain=1.0)
        nn.init.zeros_(out.bias)
        lins.append(out)
        self.linears = nn.ModuleList(lins)
        self.activations = nn.ModuleList(acts)

    def forward(self, x, y):
        z = torch.cat([2.0*x - 1.0, 2.0*y - 1.0], dim=1)
        for i in range(self.depth - 1):
            z = self.activations[i](self.linears[i](z))
        return self.linears[-1](z)

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def helmholtz_loss(model, xc, yc, fc, xb, yb, ub, w_bc=200.0):
    xc = xc.requires_grad_(True); yc = yc.requires_grad_(True)
    u = model(xc, yc)
    ux = torch.autograd.grad(u, xc, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    uxx = torch.autograd.grad(ux, xc, torch.ones_like(ux), create_graph=True, retain_graph=True)[0]
    uy = torch.autograd.grad(u, yc, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    uyy = torch.autograd.grad(uy, yc, torch.ones_like(uy), create_graph=True, retain_graph=True)[0]
    res = -(uxx + uyy) - K**2 * u - fc
    lp = torch.mean(res**2)
    lb = torch.mean((model(xb, yb) - ub)**2)
    return lp + w_bc * lb, lp.item(), lb.item()


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


def train_variant(model, label):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model.to(DEVICE)
    xc, yc, fc, xb, yb, ub = sample_points()
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-4, weight_decay=1e-6
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=4000, T_mult=2, eta_min=1e-5
    )
    t0 = time.time()
    print(f"[{label}] Adam training (trainable params: {model.count_trainable():,}) ...")
    for ep in range(1, 20001):
        opt.zero_grad()
        loss, lp, lb = helmholtz_loss(model, xc, yc, fc, xb, yb, ub)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        opt.step(); sch.step(ep)
        if ep % 4000 == 0:
            print(f"  [{label}] Ep {ep} | Loss {loss.item():.3e}")
    print(f"  [{label}] Adam done: {time.time()-t0:.0f}s")

    for rnd in range(4):
        opt_lb = torch.optim.LBFGS(
            [p for p in model.parameters() if p.requires_grad],
            lr=1.0, max_iter=500, max_eval=2000, history_size=50,
            tolerance_grad=1e-10, tolerance_change=1e-12,
            line_search_fn='strong_wolfe'
        )
        last_loss = [None]
        def closure():
            opt_lb.zero_grad()
            loss, _, _ = helmholtz_loss(model, xc, yc, fc, xb, yb, ub)
            loss.backward(); last_loss[0] = loss.item(); return loss
        opt_lb.step(closure)
        print(f"  [{label}] LBFGS round {rnd+1} | final={last_loss[0]:.4e}")

    total_t = time.time() - t0
    print(f"  [{label}] TOTAL {total_t:.0f}s | final loss={last_loss[0]:.4e}")
    return last_loss[0], total_t


def evaluate(model, xg, yg, batch=4096):
    model.eval()
    Xm, Ym = np.meshgrid(xg, yg); xf = Xm.flatten(); yf = Ym.flatten()
    n = len(xf); uf = np.zeros(n)
    with torch.no_grad():
        for s in range(0, n, batch):
            e = min(s + batch, n)
            xt = torch.tensor(xf[s:e], dtype=torch.float32, device=DEVICE).reshape(-1, 1)
            yt = torch.tensor(yf[s:e], dtype=torch.float32, device=DEVICE).reshape(-1, 1)
            uf[s:e] = model(xt, yt).cpu().numpy().flatten()
    return uf.reshape(len(yg), len(xg))


def compute_metrics(pred, ref):
    d = pred.flatten() - ref.flatten(); r = ref.flatten()
    return dict(
        L2=np.linalg.norm(d) / (np.linalg.norm(r) + 1e-12),
        L1=np.sum(np.abs(d)) / (np.sum(np.abs(r)) + 1e-12),
        Linf=np.max(np.abs(d))
    )


def get_learned_params(model):
    a_vals, b_vals, g_vals = [], [], []
    with torch.no_grad():
        for act in model.activations:
            a_vals.extend(torch.clamp(act.alpha, 0.0, 1.0).cpu().numpy().tolist())
            b_vals.extend(act.beta.cpu().numpy().tolist())
            g_vals.extend(act.gamma.cpu().numpy().tolist())
    return (float(np.mean(a_vals)), float(np.mean(b_vals)), float(np.mean(g_vals)))


# ── DEFINE ABLATION VARIANTS ─────────────────────────────────────────
variants = {
    'V1_fixed_gamma':      dict(alpha_init=0.5, beta_init=1.0, gamma_init=np.pi,
                                freeze_alpha=False, freeze_beta=False, freeze_gamma=True),
    'V2_fixed_alpha':      dict(alpha_init=0.5, beta_init=1.0, gamma_init=np.pi,
                                freeze_alpha=True,  freeze_beta=False, freeze_gamma=False),
    'V3_learnable_beta_only': dict(alpha_init=0.5, beta_init=1.0, gamma_init=np.pi,
                                freeze_alpha=True,  freeze_beta=False, freeze_gamma=True),
    'V4_pure_gabor':       dict(alpha_init=0.0, beta_init=1.0, gamma_init=np.pi,
                                freeze_alpha=True,  freeze_beta=False, freeze_gamma=False),
}

# ── RUN ALL VARIANTS ─────────────────────────────────────────────────
xg, yg, Uex, Ufdm = compute_fdm()

results = {}
for label, cfg in variants.items():
    print("\n" + "="*70)
    print(f"  RUNNING: {label}")
    print(f"  config : {cfg}")
    print("="*70)
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = AblationPINN(cfg)
    final_loss, total_t = train_variant(model, label)
    Upred = evaluate(model, xg, yg)
    met = compute_metrics(Upred, Uex)
    a_bar, b_bar, g_bar = get_learned_params(model)
    results[label] = dict(
        L2=met['L2'], L1=met['L1'], Linf=met['Linf'],
        loss=final_loss, time=total_t,
        alpha=a_bar, beta=b_bar, gamma=g_bar,
        trainable_params=model.count_trainable()
    )
    print(f"  [{label}] L2={met['L2']*100:.4f}% | L1={met['L1']*100:.4f}% | "
          f"Linf={met['Linf']:.4e}")
    print(f"  [{label}] Learned means: α={a_bar:.4f}, β={b_bar:.4f}, γ={g_bar:.4f}")

# ── SUMMARY TABLE ────────────────────────────────────────────────────
print("\n" + "="*70)
print("  ABLATION STUDY RESULTS — 2D Helmholtz")
print("="*70)
print("  Full LSSA reference (from finalized manuscript):")
print(f"    L2=0.0027% | L1=0.0018% | γ̄=2.9717 | trainable=68,481")
print("-"*70)
print(f"  {'Variant':<25} {'L2 (%)':>10} {'L1 (%)':>10} {'γ̄':>8} "
      f"{'β̄':>8} {'ᾱ':>8}")
print("-"*70)
for label in variants:
    r = results[label]
    print(f"  {label:<25} {r['L2']*100:>10.4f} {r['L1']*100:>10.4f} "
          f"{r['gamma']:>8.4f} {r['beta']:>8.4f} {r['alpha']:>8.4f}")
print("="*70)

# ── LaTeX-READY TABLE ROWS ───────────────────────────────────────────
print("\nLaTeX table rows (for insertion into ablation table):")
print("-"*70)
for label in variants:
    r = results[label]
    pretty = {
        'V1_fixed_gamma':          'Fixed $\\gamma$',
        'V2_fixed_alpha':          'Fixed $\\alpha$',
        'V3_learnable_beta_only':  'Learnable $\\beta$ only',
        'V4_pure_gabor':           'Pure Gabor',
    }[label]
    print(f"  {pretty:<26} & {r['L2']*100:.4f} & {r['L1']*100:.4f} & "
          f"${r['gamma']:.3f}$ & ${r['beta']:.3f}$ & ${r['alpha']:.3f}$ \\\\")
print("  \\midrule")
print(f"  Full LSSA (ref)            & 0.0027 & 0.0018 & $2.972$ & $0.921$ & $0.548$ \\\\")
