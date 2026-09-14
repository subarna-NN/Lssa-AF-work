"""
=============================================================================
SPECTRAL BASELINES COMPARISON — 2D Helmholtz Equation
Reviewer 1 request: direct numerical comparison with published spectral
methods (FF-PINN, SIREN, GaborPINN)
=============================================================================
PURPOSE:
  Head-to-head comparison of LSSA against three published spectral methods
  on the same Helmholtz benchmark, same protocol, same seed.

  Method 1: SIREN         — Sitzmann et al. NeurIPS 2020
                            activation: sin(omega0 * z)
                            init: uniform(-sqrt(6/n)/omega0, +sqrt(6/n)/omega0)
                                  first layer: uniform(-1/n, +1/n)
                            omega0 = 6.0 (chosen for low-k Helmholtz)

  Method 2: FF-PINN       — Tancik et al. NeurIPS 2020
                            input encoding: [cos(2*pi*B*x), sin(2*pi*B*x)]
                            B ~ N(0, sigma^2 * I),  sigma = 1.0
                            followed by standard tanh MLP

  Method 3: GaborPINN     — Huang & Alkhalifah IEEE GRSL 2023
                            architecture: Multiplicative Filter Network (MFN)
                            g_i(x) = exp(-gamma_i * ||x - mu_i||^2)
                                     * sin(omega_i * x + phi_i)
                            omega scale matched to Helmholtz frequency pi

  Reference: Full LSSA (from finalized manuscript, single seed 42)
             L2=0.0027%, L1=0.0018%, Linf=1.232e-4

EVERYTHING ELSE IS IDENTICAL to the finalized lssa_helmholtz_v2:
  PDE:   -(u_xx + u_yy) - k^2*u = f,  (x,y) in [0,1]^2
  BC:    u = 0 on all four edges
  Sol:   u* = sin(pi*x)*sin(pi*y),  k=1
  FDM:   5-point stencil, N=256
  Depth: 5 hidden layers, width 128
  Adam:  20000 epochs, CosineAnnealingWarmRestarts T0=4000, T_mult=2
  LBFGS: 4 rounds x 500 iters
  w_bc=200, N_col=10000, N_bc=800, seed=42

OUTPUT: text-only results — per-method L2, L1, Linf, final loss, wall
        time, and parameter count. Ready for direct table insertion.
=============================================================================
"""

import torch, torch.nn as nn
import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve
import time, warnings, math
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


# =====================================================================
# METHOD 1: SIREN
# Sitzmann, Martel, Bergman, Lindell, Wetzstein.
# Implicit Neural Representations with Periodic Activation Functions.
# NeurIPS 2020.
# =====================================================================
class SIREN(nn.Module):
    """
    5x128 MLP with sin(omega0 * z) activations and SIREN-specific init.
    First-layer weights ~ U(-1/n_in, +1/n_in).
    Hidden-layer weights ~ U(-sqrt(6/n_in)/omega0, +sqrt(6/n_in)/omega0).
    Output layer: standard Xavier init, no activation.
    omega0 = 6.0 chosen for low-k Helmholtz (k=1, target freq = pi ~ 3.14).
    """
    def __init__(self, omega0=6.0, layers=None):
        super().__init__()
        if layers is None: layers = [2, 128, 128, 128, 128, 128, 1]
        self.omega0 = omega0
        self.depth = len(layers) - 1
        lins = []
        for i in range(self.depth):
            l = nn.Linear(layers[i], layers[i+1])
            n_in = layers[i]
            with torch.no_grad():
                if i == 0:
                    l.weight.uniform_(-1.0/n_in, 1.0/n_in)
                elif i < self.depth - 1:
                    bound = math.sqrt(6.0/n_in) / omega0
                    l.weight.uniform_(-bound, bound)
                else:
                    nn.init.xavier_uniform_(l.weight, gain=1.0)
                l.bias.zero_()
            lins.append(l)
        self.linears = nn.ModuleList(lins)

    def forward(self, x, y):
        z = torch.cat([2.0*x - 1.0, 2.0*y - 1.0], dim=1)
        for i in range(self.depth - 1):
            z = torch.sin(self.omega0 * self.linears[i](z))
        return self.linears[-1](z)


# =====================================================================
# METHOD 2: Fourier Features PINN (FF-PINN)
# Tancik, Srinivasan, Mildenhall, Fridovich-Keil, Raghavan, Singhal,
# Ramamoorthi, Barron, Ng.
# Fourier Features Let Networks Learn High Frequency Functions
# in Low Dimensional Domains. NeurIPS 2020.
# =====================================================================
class FFPINN(nn.Module):
    """
    Gaussian random Fourier feature encoding followed by tanh MLP.
    Encoding:  gamma(x) = [cos(2*pi*B*x), sin(2*pi*B*x)]
               B ~ N(0, sigma^2 * I),  B shape: (m, 2)
    sigma = 1.0 (Tancik et al. recommend for low-frequency tasks; here k=1).
    m = 128 (encoding dimension per component -> 256 features total).
    Followed by 5-hidden-layer MLP with tanh, width 128, matching depth.
    """
    def __init__(self, sigma=1.0, m=128, layers=None):
        super().__init__()
        if layers is None:
            layers = [2*m, 128, 128, 128, 128, 128, 1]
        self.sigma = sigma; self.m = m
        # random encoding matrix B (fixed, NOT learned - Tancik et al.)
        B = torch.randn(m, 2) * sigma
        self.register_buffer('B', B)
        self.depth = len(layers) - 1
        lins = []
        for i in range(self.depth):
            l = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_uniform_(l.weight, gain=1.0)
            nn.init.zeros_(l.bias)
            lins.append(l)
        self.linears = nn.ModuleList(lins)

    def forward(self, x, y):
        xy = torch.cat([2.0*x - 1.0, 2.0*y - 1.0], dim=1)   # (batch, 2)
        proj = 2.0 * math.pi * xy @ self.B.T                 # (batch, m)
        z = torch.cat([torch.cos(proj), torch.sin(proj)], dim=1)  # (batch, 2m)
        for i in range(self.depth - 1):
            z = torch.tanh(self.linears[i](z))
        return self.linears[-1](z)


# =====================================================================
# METHOD 3: GaborPINN
# Huang, Alkhalifah.
# GaborPINN: Efficient Physics Informed Neural Networks Using
# Multiplicative Filtered Networks.
# IEEE Geoscience and Remote Sensing Letters, 2023.
#
# Built on the Multiplicative Filter Network (Fathony et al. ICLR 2021).
# Architecture:
#   z_1 = g_1(x)
#   z_{i+1} = (W_i z_i + b_i) * g_{i+1}(x)     (elementwise *)
#   output = W_k z_k + b_k
# with Gabor filters:
#   g_i(x) = exp(-0.5 * gamma_i * ||x - mu_i||^2) * sin(omega_i^T x + phi_i)
# Parameters gamma_i, mu_i, omega_i, phi_i are all learnable.
# omega initialized with scale matched to Helmholtz frequency pi.
# =====================================================================
class GaborFilter(nn.Module):
    """One Gabor filter block producing an (output_dim,)-vector for each
    input point x in R^input_dim."""
    def __init__(self, input_dim, output_dim, freq_scale=np.pi,
                 alpha_scale=1.0):
        super().__init__()
        self.mu    = nn.Parameter(torch.rand(output_dim, input_dim)*2.0 - 1.0)
        self.gamma = nn.Parameter(torch.ones(output_dim) * alpha_scale)
        self.omega = nn.Parameter(torch.randn(output_dim, input_dim) * freq_scale)
        self.phi   = nn.Parameter(torch.rand(output_dim) * 2.0 * np.pi)

    def forward(self, x):
        # x: (batch, input_dim); mu: (output_dim, input_dim)
        diff = x.unsqueeze(1) - self.mu.unsqueeze(0)              # (b, out, in)
        env  = torch.exp(-0.5 * self.gamma * (diff**2).sum(dim=-1))  # (b, out)
        proj = x @ self.omega.T + self.phi                        # (b, out)
        return env * torch.sin(proj)


class GaborPINN(nn.Module):
    """
    Multiplicative Filter Network with Gabor filters at every layer.
    Depth: 5 hidden layers, width 128 (matches other methods).
    Freq scale for omega: pi (Helmholtz solution frequency).
    Alpha (Gaussian window scale) initialized to 1.0.
    """
    def __init__(self, input_dim=2, hidden=128, depth=5,
                 freq_scale=np.pi, alpha_scale=1.0):
        super().__init__()
        self.depth = depth
        # k=depth Gabor filters, one per layer
        self.filters = nn.ModuleList([
            GaborFilter(input_dim, hidden, freq_scale, alpha_scale)
            for _ in range(depth)
        ])
        # depth-1 linear maps between filter stages
        self.linears = nn.ModuleList([
            nn.Linear(hidden, hidden) for _ in range(depth - 1)
        ])
        for l in self.linears:
            nn.init.xavier_uniform_(l.weight, gain=1.0)
            nn.init.zeros_(l.bias)
        # output projection
        self.output = nn.Linear(hidden, 1)
        nn.init.xavier_uniform_(self.output.weight, gain=1.0)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, y):
        xy = torch.cat([2.0*x - 1.0, 2.0*y - 1.0], dim=1)
        z = self.filters[0](xy)                                # (b, hidden)
        for i in range(1, self.depth):
            z = self.linears[i-1](z) * self.filters[i](xy)     # MFN update
        return self.output(z)


# ── LOSS AND SAMPLING (shared across all three methods) ──────────────
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


def train_method(model, label):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model.to(DEVICE)
    xc, yc, fc, xb, yb, ub = sample_points()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{label}] Adam training (trainable params: {n_params:,}) ...")
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=4000, T_mult=2, eta_min=1e-5
    )
    t0 = time.time()
    for ep in range(1, 20001):
        opt.zero_grad()
        loss, lp, lb = helmholtz_loss(model, xc, yc, fc, xb, yb, ub)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(ep)
        if ep % 4000 == 0:
            print(f"  [{label}] Ep {ep} | Loss {loss.item():.3e}")
    print(f"  [{label}] Adam done: {time.time()-t0:.0f}s")

    for rnd in range(4):
        opt_lb = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=500, max_eval=2000,
            history_size=50, tolerance_grad=1e-10, tolerance_change=1e-12,
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
    return last_loss[0], total_t, n_params


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


# ── RUN ALL THREE BASELINES ──────────────────────────────────────────
xg, yg, Uex, Ufdm = compute_fdm()

methods = [
    ('SIREN (omega0=6.0)',     lambda: SIREN(omega0=6.0)),
    ('FF-PINN (sigma=1.0)',    lambda: FFPINN(sigma=1.0, m=128)),
    ('GaborPINN',              lambda: GaborPINN(input_dim=2, hidden=128,
                                                  depth=5, freq_scale=np.pi,
                                                  alpha_scale=1.0)),
]

results = {}
for label, ctor in methods:
    print("\n" + "="*70)
    print(f"  RUNNING: {label}")
    print("="*70)
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = ctor()
    final_loss, total_t, n_params = train_method(model, label)
    Upred = evaluate(model, xg, yg)
    met = compute_metrics(Upred, Uex)
    results[label] = dict(
        L2=met['L2'], L1=met['L1'], Linf=met['Linf'],
        loss=final_loss, time=total_t, params=n_params
    )
    print(f"  [{label}] L2={met['L2']*100:.4f}% | L1={met['L1']*100:.4f}% | "
          f"Linf={met['Linf']:.4e}")

# ── SUMMARY TABLE ────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SPECTRAL BASELINE COMPARISON — 2D Helmholtz")
print("="*70)
print("  LSSA reference (from finalized manuscript, seed 42):")
print(f"    L2=0.0027% | L1=0.0018% | Linf=1.232e-04 | 68,481 params | 3221s")
print("-"*70)
print(f"  {'Method':<25} {'L2 (%)':>10} {'L1 (%)':>10} {'Linf':>12} "
      f"{'Params':>10} {'Time(s)':>8}")
print("-"*70)
for label in [m[0] for m in methods]:
    r = results[label]
    print(f"  {label:<25} {r['L2']*100:>10.4f} {r['L1']*100:>10.4f} "
          f"{r['Linf']:>12.4e} {r['params']:>10,} {r['time']:>8.0f}")
print("="*70)

# ── LaTeX-READY TABLE ROWS ───────────────────────────────────────────
print("\nLaTeX table rows:")
print("-"*70)
for label in [m[0] for m in methods]:
    r = results[label]
    pretty = {
        'SIREN (omega0=6.0)':  'SIREN~\\cite{sitzmann2020implicit}',
        'FF-PINN (sigma=1.0)': 'FF-PINN~\\cite{tancik2020fourier}',
        'GaborPINN':           'GaborPINN~\\cite{huang2023gaborpinn}',
    }[label]
    print(f"  {pretty:<40} & {r['L2']*100:.4f} & {r['L1']*100:.4f} & "
          f"${r['Linf']:.3e}$ & {r['params']:,} \\\\")
print("  \\midrule")
print(f"  \\textbf{{LSSA (proposed)}}                & \\textbf{{0.0027}} & "
      f"\\textbf{{0.0018}} & $\\bm{{1.232\\!\\times\\!10^{{-4}}}}$ & \\textbf{{68,481}} \\\\")
