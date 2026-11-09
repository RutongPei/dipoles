"""Does the fit-minus-direct gap in m12i equal the leakage term mu_r<r_hat> + mu_th<th_hat> + mu_ph<ph_hat>?
Also flags fits that sit at a parameter bound (a sign the optimizer did not find a clean optimum).

Run on SCC after:  python scc/04_direct_m12i.py --weights number   (fit is number-weighted too)
    python scc/06_check_leak_m12i.py                       # all centres, both selections
    python scc/06_check_leak_m12i.py --centres 150,160,170  # just these
"""
import argparse
import csv, os, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--centres', default='')
args = ap.parse_args()
CENTRES = [int(c) for c in args.centres.split(',')] if args.centres else list(range(30, 301, 10))
D = os.environ.get('DIPOLE_DATA_DIR', '/projectnb/fire-reflex/rpei/data')
d = np.load(f'{D}/m12i-600-halostars.npz'); xyz = d['xyz']; dist = d['dist']
masks = {'none': np.ones(len(dist), bool), 'bound_alex': d['mask_bound_alex']}
def basis(x):
    r = np.linalg.norm(x, axis=1); th = np.arccos(x[:, 2] / r); ph = np.arctan2(x[:, 1], x[:, 0])
    return (x / r[:, None],
            np.stack([np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), -np.sin(th)], 1),
            np.stack([-np.sin(ph), np.cos(ph), 0*ph], 1))
rows = lambda f: {(r['mask'], r['selection'], int(float(r['center_kpc']))): r for r in csv.DictReader(open(f))}
fit, dirn = rows(f'{D}/m12i-600-fits.csv'), rows(f'{D}/m12i-600-direct-number.csv')
vec = lambda r, k: np.array([float(r[k + c]) for c in 'xyz'])
for sel_name in ['binned', 'complementary']:
  print(f'\n===== {sel_name} =====')
  for m in ['none', 'bound_alex']:
    for c in CENTRES:
        f, g = fit[(m, sel_name, c)], dirn[(m, sel_name, c)]
        sel = masks[m] & (((dist > c - 2.5) & (dist < c + 2.5)) if sel_name == 'binned' else ((dist > c) & (dist < 305)))
        a, b, cc = basis(xyz[sel])
        mu = [float(f['vr_mean']), float(f['vtheta_mean']), float(f['vphi_mean'])]
        leak = mu[0]*a.mean(0) + mu[1]*b.mean(0) + mu[2]*cc.mean(0)
        gap = vec(f, 'v') - vec(g, 'v')
        print(f'{m:10s} {c:3d}  |fit|={np.linalg.norm(vec(f,"v")):5.1f} |direct|={np.linalg.norm(vec(g,"v")):5.1f}  '
              f'gap={np.round(gap,1)}  leak={np.round(leak,1)}  mu={np.round(mu,0)}  |<r_hat>|={np.linalg.norm(a.mean(0)):.2f}')
        sig = [float(f[k]) for k in ['sigma_r', 'sigma_theta', 'sigma_phi']]
        at = [n for n, val, b in [('v', np.abs(vec(f, 'v')).max(), 300), ('mean', np.abs(mu).max(), 500)] if val > 0.99 * b]
        at += ['sigma'] if (min(sig) < 0.011 or max(sig) > 0.99 * 10**2.5) else []
        print(f'{"":14s} sigma={np.round(sig, 1)}  converged={f["converged"]}' + (f'  AT BOUND: {at}' if at else ''))
