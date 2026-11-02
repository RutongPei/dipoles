"""Direct (model-free) v_travel for m12i snapshot 600 from the saved halo stars.

Replaces the 9-parameter likelihood fit (03_fit_m12i.py) with the quantity the fit
is trying to estimate: the velocity of the host's inner reference frame relative to
the centre of mass of the selected outer stars,

    v_travel_vec = v_ref - <v>_sel = -( sum_i m_i v_i ) / ( sum_i m_i ),

because v_xyz in the npz is already relative to the host velocity (v_ref = 0 there).
The sign matches the fit's pars[:3] (dipole.generate_fake_data builds data as
v = intrinsic - pars[:3]), so the vx, vy, vz columns are directly comparable with
m12i-600-fits.csv.

No optimizer, no assumed velocity distribution, no constant-mean-per-shell model.
Uncertainty = particle shot noise of a weighted mean (analytic).

Same selections and masks as 03_fit_m12i.py. Needs only numpy (+ matplotlib for the plot).

    python scc/04_direct_m12i.py                    # mass-weighted (momentum) definition
    python scc/04_direct_m12i.py --weights number   # number-weighted (what star counts give)

Outputs (in $DIPOLE_DATA_DIR, default /projectnb/fire-reflex/rpei/data):
    m12i-600-direct.csv
    m12i-600-direct.png   direct |v_travel| vs distance; fit overlaid if m12i-600-fits.csv exists
"""
import argparse
import csv
import os

import numpy as np


def direct_v_travel(v_xyz, weights=None, xyz=None):
    """Model-free reflex vector of a particle selection.

    Parameters
    ----------
    v_xyz   : (N, 3) velocities in the host frame [km/s]
    weights : (N,) particle masses (momentum definition) or None (number-weighted)
    xyz     : (N, 3) positions, optional; only used for the sky-lopsidedness diagnostic

    Returns dict with
    vec      : (3,) v_travel vector, same sign convention as the fit's pars[:3]
    err      : (3,) 1-sigma shot-noise error per component (weighted standard error)
    n, n_eff : particle count and effective count (sum w)^2 / sum w^2
    floor    : expected |vec| if the true reflex were zero (shot noise only)
    lopsided : |weighted mean of r_hat|; 0 for an even sky distribution, -> 1 for one clump.
               When this is large, net radial flow or rotation of the lopsided stars
               contributes to vec (real momentum, but not reflex in the usual sense).
    """
    v = np.asarray(v_xyz, dtype=np.float64)
    n = len(v)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    W = w.sum()
    mean_v = (w[:, None] * v).sum(axis=0) / W
    resid = v - mean_v
    err = np.sqrt((w[:, None]**2 * resid**2).sum(axis=0)) / W
    n_eff = W**2 / np.sum(w**2)
    # E|X| for X ~ N(0, diag(err^2)); exact for isotropic err, close otherwise
    floor = np.sqrt(8 / (3 * np.pi)) * np.sqrt(np.sum(err**2))
    out = dict(vec=-mean_v, err=err, n=n, n_eff=n_eff, floor=floor, lopsided=np.nan)
    if xyz is not None:
        x = np.asarray(xyz, dtype=np.float64)
        rhat = x / np.linalg.norm(x, axis=1)[:, None]
        out['lopsided'] = np.linalg.norm((w[:, None] * rhat).sum(axis=0) / W)
    return out


def vec_to_angles(vec):
    v = np.linalg.norm(vec)
    if v == 0:
        return v, np.nan, np.nan
    theta = np.degrees(np.arccos(vec[2] / v))          # polar angle from disc normal
    phi = np.degrees(np.arctan2(vec[1], vec[0])) % 360
    return v, theta, phi


CENTERS = np.arange(30, 301, 10)
WIDTH, HIGH = 5, 305


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', default='m12i')
    ap.add_argument('--snap', type=int, default=600)
    ap.add_argument('--masks', default='none,bound_alex,bound_rockstar')
    ap.add_argument('--weights', choices=['mass', 'number'], default='mass')
    ap.add_argument('--data-dir', default=os.environ.get('DIPOLE_DATA_DIR',
                                                         '/projectnb/fire-reflex/rpei/data'))
    args = ap.parse_args()

    tag = f'{args.sim}-{args.snap}'
    d = np.load(os.path.join(args.data_dir, f'{tag}-halostars.npz'))
    xyz = d['xyz'].astype(np.float64)
    v_xyz = d['v_xyz'].astype(np.float64)
    dist = d['dist'].astype(np.float64)
    mass = d['mass'].astype(np.float64) if args.weights == 'mass' else None
    masks = {'none': np.ones(len(dist), bool)}
    masks.update({k[5:]: d[k] for k in d.files if k.startswith('mask_')})
    mask_names = [m for m in args.masks.split(',') if m in masks]

    def selection(analysis, c):
        if analysis == 'binned':
            return (dist > c - WIDTH / 2) & (dist < c + WIDTH / 2)
        return (dist > c) & (dist < HIGH)               # Alex's 'complementary'

    header = ['mask', 'selection', 'center_kpc', 'N', 'N_eff', 'v_travel', 'theta_deg', 'phi_deg',
              'shot_noise_floor', 'vx', 'vy', 'vz', 'err_vx', 'err_vy', 'err_vz', 'lopsided']
    rows = []
    for m in mask_names:
        for analysis in ['binned', 'complementary']:
            for c in CENTERS:
                sel = masks[m] & selection(analysis, c)
                if sel.sum() < 2:
                    rows.append([m, analysis, int(c), int(sel.sum())] + [np.nan] * 12)
                    continue
                r = direct_v_travel(v_xyz[sel], None if mass is None else mass[sel], xyz[sel])
                v, th, ph = vec_to_angles(r['vec'])
                rows.append([m, analysis, int(c), r['n'], r['n_eff'], v, th, ph, r['floor'],
                             *r['vec'], *r['err'], r['lopsided']])
                print(f'{m:15s} {analysis:13s} {c:3d} kpc  N = {r["n"]:8,d}  '
                      f'|v_travel| = {v:5.1f} +- floor {r["floor"]:4.1f} km/s  '
                      f'(theta {th:5.1f}, phi {ph:5.1f})  lopsided {r["lopsided"]:.2f}', flush=True)

    suffix = '' if args.weights == 'mass' else '-number'
    out_csv = os.path.join(args.data_dir, f'{tag}-direct{suffix}.csv')
    with open(out_csv, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(header)
        for row in rows:
            wr.writerow([f'{x:.4g}' if isinstance(x, float) else x for x in row])
    print('saved', out_csv)

    # ---- plot, with the likelihood fit overlaid if available ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fits = {}
    fit_csv = os.path.join(args.data_dir, f'{tag}-fits.csv')
    if os.path.exists(fit_csv):
        with open(fit_csv) as f:
            for r in csv.DictReader(f):
                fits.setdefault((r['mask'], r['selection']), []).append(
                    (float(r['center_kpc']), float(r['v_travel'])))

    colors = {'none': '0.55', 'bound_rockstar': '#d95f02', 'bound_alex': '#1b6ac9'}
    labels = {'none': 'all stars', 'bound_rockstar': 'bound removed (Rockstar)',
              'bound_alex': "bound removed (Alex's assignment)"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, analysis, title in zip(axes, ['binned', 'complementary'],
                                   ['binned (5 kpc shells)', f'complementary (r > centre, < {HIGH} kpc)']):
        for m in mask_names:
            rr = [r for r in rows if r[0] == m and r[1] == analysis]
            c = np.array([r[2] for r in rr], float)
            v = np.array([r[5] for r in rr], float)
            fl = np.array([r[8] for r in rr], float)
            col = colors.get(m, 'k')
            ax.plot(c, v, marker='o', ms=3, lw=1.5, color=col, label=labels.get(m, m) + ' (direct)')
            ax.plot(c, fl, ls=':', lw=1.2, color=col)
            if (m, analysis) in fits:
                cf, vf = np.array(fits[(m, analysis)]).T
                ax.plot(cf, vf, ls='--', lw=1, color=col, alpha=0.7, label=labels.get(m, m) + ' (fit)')
        ax.set_title(title)
        ax.set_xlabel('distance from host centre [kpc]')
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r'$|v_{\rm travel}|$ [km/s]')
    axes[0].plot([], [], ls=':', color='k', label='dotted: shot-noise floor')
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f'{args.sim} snapshot {args.snap}: direct v_travel '
                 f'({args.weights}-weighted COM of selected stars, host frame)', fontsize=11)
    fig.tight_layout()
    out_png = os.path.join(args.data_dir, f'{tag}-direct{suffix}.png')
    fig.savefig(out_png, dpi=150)
    print('saved', out_png)


if __name__ == '__main__':
    main()
