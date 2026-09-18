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
print("="*70)

# ── EXACT AND SOURCE ─────────────────────────────────────────────────
def u_exact(x, y):  return np.sin(np.pi*x) * np.sin(np.pi*y)
def f_source(x, y): return F_AMP * np.sin(np.pi*x) * np.sin(np.pi*y)

# ── FDM REFERENCE  ───────────────────
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

# ── LSSA ACTIVATION ──────────────────────────────────────────────────
class LSSAActivation(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((n,), 0.5))
        self.beta  = nn.Parameter(torch.ones(n))
        self.gamma = nn.Parameter(torch.full((n,), float(np.pi)))
    def forward(self, z):
        a = torch.clamp(self.alpha, 0.01, 0.99).unsqueeze(0)
        b = self.beta.unsqueeze(0); g = self.gamma.unsqueeze(0)
        return a*torch.tanh(b*z) + (1-a)*torch.sin(g*z)*torch.exp(-0.5*z**2)

# ── UNIFIED PINN ────────────
class ConfigurablePINN(nn.Module):
    def __init__(self, method, depth, width):
        super().__init__()
        # layers: [2, width, width, ..., width, 1]  with `depth` hidden layers
        layers = [2] + [width]*depth + [1]
        self.n_layers = len(layers) - 1
        self.method = method
        lins, acts = [], []
        for i in range(self.n_layers - 1):
            l = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_uniform_(l.weight, gain=1.0)
            nn.init.zeros_(l.bias)
            lins.append(l)
            if method == 'lssa':
                acts.append(LSSAActivation(layers[i+1]))
        out = nn.Linear(layers[-2], layers[-1])
        nn.init.xavier_uniform_(out.weight, gain=1.0)
        nn.init.zeros_(out.bias)
        lins.append(out)
        self.linears = nn.ModuleList(lins)
        self.activations = nn.ModuleList(acts) if method == 'lssa' else None

    def forward(self, x, y):
        z = torch.cat([2.0*x - 1.0, 2.0*y - 1.0], dim=1)
        for i in range(self.n_layers - 1):
            pre = self.linears[i](z)
            z = self.activations[i](pre) if self.method == 'lssa' else torch.tanh(pre)
        return self.linears[-1](z)

    def count_params(self):
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


def train(model, label):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model.to(DEVICE)
    xc, yc, fc, xb, yb, ub = sample_points()
    npar = model.count_params()
    print(f"[{label}] training (params: {npar:,}) ...")
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=4000, T_mult=2, eta_min=1e-5)
    t0 = time.time()
    for ep in range(1, 20001):
        opt.zero_grad()
        loss, lp, lb = helmholtz_loss(model, xc, yc, fc, xb, yb, ub)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(ep)
        if ep % 5000 == 0:
            print(f"  [{label}] Ep {ep} | Loss {loss.item():.3e}")
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
    total_t = time.time() - t0
    print(f"  [{label}] done {total_t:.0f}s | final loss={ll[0]:.4e}")
    return npar, total_t


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


# ── ARCHITECTURE CONFIGURATIONS ──────────────────────────────────────
configs = [
    ('A', 3, 64),
    ('B', 3, 128),
    ('C', 5, 128),   # paper default
    ('D', 7, 128),
]

xg, yg, Uex, Ufdm = compute_fdm()

results = {}
for cid, depth, width in configs:
    for method in ['lssa', 'tanh']:
        label = f"{cid}_{method}_{depth}x{width}"
        print("\n" + "="*70)
        print(f"  CONFIG {cid}: {depth}x{width} | METHOD: {method.upper()}")
        print("="*70)
        torch.manual_seed(SEED); np.random.seed(SEED)
        model = ConfigurablePINN(method, depth, width)
        npar, total_t = train(model, label)
        Upred = evaluate(model, xg, yg)
        met = compute_metrics(Upred, Uex)
        results[(cid, method)] = dict(
            depth=depth, width=width, L2=met['L2'], L1=met['L1'],
            Linf=met['Linf'], params=npar, time=total_t
        )
        print(f"  [{label}] L2={met['L2']*100:.4f}% | L1={met['L1']*100:.4f}% | "
              f"Linf={met['Linf']:.4e}")

# ── SUMMARY TABLE ────────────────────────────────────────────────────
print("\n" + "="*70)
print("  ARCHITECTURE ABLATION — 2D Helmholtz — LSSA vs tanh")
print("="*70)
print(f"  {'Config':<8}{'Arch':<10}{'Method':<8}{'L2 (%)':>10}{'L1 (%)':>10}"
      f"{'Params':>10}")
print("-"*70)
for cid, depth, width in configs:
    for method in ['lssa', 'tanh']:
        r = results[(cid, method)]
        print(f"  {cid:<8}{f'{depth}x{width}':<10}{method.upper():<8}"
              f"{r['L2']*100:>10.4f}{r['L1']*100:>10.4f}{r['params']:>10,}")
    print("-"*70)

# ── PAIRED COMPARISON (LSSA advantage factor) ────────────────────────
print("\n  LSSA advantage over tanh (L2 ratio) at each architecture:")
for cid, depth, width in configs:
    rl = results[(cid, 'lssa')]; rt = results[(cid, 'tanh')]
    ratio = rt['L2'] / rl['L2'] if rl['L2'] > 0 else float('inf')
    print(f"    Config {cid} ({depth}x{width}): tanh/LSSA L2 = {ratio:.1f}x  "
          f"(LSSA {'wins' if rl['L2'] < rt['L2'] else 'loses'})")
