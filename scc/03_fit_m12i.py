"""Step 6: dipole fits for m12i snapshot 600 from the saved halo stars, plus the plot.

Uses Alex's model and optimizer from dipole.py, his distance selections
(binned: 5 kpc shells centred 30-300 kpc; complementary: c < r < 305 kpc),
and the bound-star masks saved by 02_read_m12i_600.py. No gizmo/snapshot reading.

Run on a compute node from the repo top level:
    python scc/03_fit_m12i.py                 # local optimizer (fast; see note)
    python scc/03_fit_m12i.py --global 20     # Alex's basinhopping, 20 hops (slow)

Note on the optimizer: Alex used basinhopping (repeated local searches from random
starts). Default here is one local SLSQP search on the per-star mean log-likelihood,
which on mock data reaches the same optimum as his basinhopping (see fit_local_per_star).
Run --global on a subset if you want to confirm on the real data.

Outputs (in $DIPOLE_DATA_DIR, default /projectnb/fire-reflex/rpei/data):
    m12i-600-fits.csv   one row per (mask, selection, centre)
    m12i-600-dipole.png v_travel vs distance, with the Poisson-only floor
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
import dipole  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--sim', default='m12i')
ap.add_argument('--snap', type=int, default=600)
ap.add_argument('--global', dest='niter', type=int, default=0,
                help='use Alex\'s basinhopping with this many hops (he used 100)')
ap.add_argument('--masks', default='none,bound_alex,bound_rockstar')
ap.add_argument('--nproc', type=int, default=int(os.environ.get('NSLOTS', 1)))
args = ap.parse_args()

data_dir = dipole.get_data_directory()
d = np.load(os.path.join(data_dir, f'{args.sim}-{args.snap}-halostars.npz'))
xyz = d['xyz'].astype(np.float64)
v_sph = d['v_sph'].astype(np.float64)
dist = d['dist'].astype(np.float64)
masks = {'none': np.ones(len(dist), bool)}
masks.update({k[5:]: d[k] for k in d.files if k.startswith('mask_')})
mask_names = [m for m in args.masks.split(',') if m in masks]

CENTERS = np.arange(30, 301, 10)
WIDTH, HIGH = 5, 305
INITIAL = [0, 0, 0, 0, 0, 0, np.log10(50), np.log10(50), np.log10(50)]

# Poisson-only floor from fake_data_experiments.py (true v_travel = 0, sigma = 100, 80, 80)
N_TAB = np.array([100, 300, 1000, 3000, 20000])
FLOOR_TAB = np.array([16., 12., 5., 3., 1.])


def floor(n):
    if n < N_TAB[0]:
        return np.nan
    if n > N_TAB[-1]:
        return FLOOR_TAB[-1] * np.sqrt(N_TAB[-1] / n)
    return 10**np.interp(np.log10(n), np.log10(N_TAB), np.log10(FLOOR_TAB))


def selection(analysis, c):
    if analysis == 'binned':
        return (dist > c - WIDTH / 2) & (dist < c + WIDTH / 2)
    return (dist > c) & (dist < HIGH)          # Alex's 'complementary'


def fit_local_per_star(x, v):
    """SLSQP on the MEAN negative log-likelihood per star.

    dipole.optimize_pars_local minimises the summed -lnL (~1e6 for 1e5 stars); SLSQP's
    absolute stopping tolerance then makes it stop near the starting point for large N
    (mock test: N = 256k returned |v_travel| = 0 instead of 28.4 km/s, while reporting
    success). Dividing by N leaves the maximum unchanged and fixes the scaling; it matches
    Alex's basinhopping result (28.1 km/s both) at a fraction of the cost.
    """
    n = len(x)
    return minimize(lambda p: -dipole.lnlike(p, x, v) / n, INITIAL, bounds=dipole.get_bounds(),
                    method='SLSQP', options={'maxiter': 500, 'ftol': 1e-10})


def fit_one(job):
    mask_name, analysis, c = job
    sel = masks[mask_name] & selection(analysis, c)
    n = int(sel.sum())
    t = time.time()
    if n < 50:
        return (mask_name, analysis, c, n, *[np.nan] * 9, False, 0.)
    if args.niter > 0:
        soln = dipole.optimize_pars_global(INITIAL, xyz[sel], v_sph[sel], niter=args.niter)
        ok = bool(soln.lowest_optimization_result.success)
    else:
        soln = fit_local_per_star(xyz[sel], v_sph[sel])
        ok = bool(soln.success)
    return (mask_name, analysis, c, n, *soln.x, ok, time.time() - t)


jobs = [(m, a, c) for m in mask_names for a in ['binned', 'complementary'] for c in CENTERS]
print(f'{len(jobs)} fits, {"basinhopping niter=%d" % args.niter if args.niter else "local SLSQP"}, '
      f'{args.nproc} process(es)', flush=True)
t0 = time.time()
with Pool(args.nproc) as pool:
    results = []
    for i, r in enumerate(pool.imap(fit_one, jobs)):
        results.append(r)
        m, a, c, n = r[:4]
        v = np.linalg.norm(r[4:7])
        print(f'[{time.time() - t0:6.0f} s] {m:15s} {a:13s} {c:3d} kpc  N = {n:7,d}  '
              f'|v_travel| = {v:5.1f} km/s (floor {floor(n):4.1f})  converged {r[13]}', flush=True)

# ---- table ----
rows = []
for r in results:
    m, a, c, n = r[:4]
    vx, vy, vz = r[4:7]
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    theta = np.degrees(np.arccos(vz / v)) if v > 0 else np.nan     # polar angle from disc normal
    phi = np.degrees(np.arctan2(vy, vx)) % 360 if v > 0 else np.nan
    rows.append([m, a, c, n, v, theta, phi, floor(n), vx, vy, vz,
                 *r[7:10], *(10**np.array(r[10:13])), r[13], r[14]])
header = ('mask,selection,center_kpc,N,v_travel,theta_deg,phi_deg,poisson_floor,'
          'vx,vy,vz,vr_mean,vtheta_mean,vphi_mean,sigma_r,sigma_theta,sigma_phi,converged,fit_seconds')
out_csv = os.path.join(data_dir, f'{args.sim}-{args.snap}-fits{"-global" if args.niter else ""}.csv')
with open(out_csv, 'w') as f:
    f.write(header + '\n')
    for row in rows:
        f.write(','.join(f'{x:.4g}' if isinstance(x, float) else str(x) for x in row) + '\n')
print('saved', out_csv)

# ---- plot ----
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

styles = {'none': dict(color='0.55', label='all stars'),
          'bound_rockstar': dict(color='#d95f02', ls='--', label='bound removed (Rockstar)'),
          'bound_alex': dict(color='#1b6ac9', label="bound removed (Alex's assignment)")}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, analysis, title in zip(axes, ['binned', 'complementary'],
                               ['binned (5 kpc shells)', f'complementary (r > centre, < {HIGH} kpc)']):
    for m in mask_names:
        rr = [r for r in rows if r[0] == m and r[1] == analysis]
        c = np.array([r[2] for r in rr])
        v = np.array([r[4] for r in rr])
        fl = np.array([r[7] for r in rr])
        ax.plot(c, v, marker='o', ms=3, lw=1.5, **styles.get(m, dict(label=m)))
        ax.plot(c, fl, ls=':', lw=1.2, color=styles.get(m, {}).get('color', 'k'), label=None)
    ax.set_title(title)
    ax.set_xlabel('distance from host centre [kpc]')
    ax.grid(alpha=0.25)
axes[0].set_ylabel(r'$|v_{\rm travel}|$ [km/s]')
axes[0].plot([], [], ls=':', color='k', label='dotted: noise-only floor at that N')
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle(f'{args.sim} snapshot {args.snap} (z = 0, no LMC): fitted reflex dipole', fontsize=11)
fig.tight_layout()
out_png = os.path.join(data_dir, f'{args.sim}-{args.snap}-dipole{"-global" if args.niter else ""}.png')
fig.savefig(out_png, dpi=150)
print('saved', out_png, f'(total {time.time() - t0:.0f} s)')
