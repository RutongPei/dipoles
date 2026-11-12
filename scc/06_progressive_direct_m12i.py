"""Progressive masking in m12i snapshot 600 with the direct v_travel estimator
(the direct-estimator version of Alex's scripts/m12i_progressive.py).

Cumulative steps, in Alex's order (a step is skipped if its mask is missing):
    all -> - bound (Alex's assignment) -> - messy stream -> - Danny streams -> - Nondh streams
For each step: direct v_travel in Alex's binned (5 kpc shells) and complementary (r > c, < 305 kpc)
selections, same as 04_direct_m12i.py.

Component breakdown. Because the direct estimate is a weighted mean, for any selection
    v_all = sum_k (W_k / W_all) * v_k        (W = total weight: mass, or number of stars)
over disjoint pieces k. The pieces are what each step removes that earlier steps did not:
    bound, messy minus bound, danny minus (bound, messy), nondh minus (...), and the rest.
c_k = (W_k / W_all) v_k is the vector each piece contributes to the all-star v_travel. Its
projection on the direction of v_all adds up exactly to |v_all|, so the bottom panels show which
pieces carry the signal and which cancel. This does not depend on the masking order.

Needs: m12i-600-halostars.npz (02) and m12i-600-streammasks.npz (05). numpy + matplotlib only.
    python scc/06_progressive_direct_m12i.py
    python scc/06_progressive_direct_m12i.py --weights number

Outputs ($DIPOLE_DATA_DIR):
    m12i-600-progressive-direct.csv             one row per step / selection / centre
    m12i-600-progressive-components.csv         one row per piece / selection / centre
    m12i-600-progressive-direct.png             |v_travel| after each step, binned and complementary
(suffix -number for number weighting)
"""
import argparse
import csv
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('direct04', os.path.join(HERE, '04_direct_m12i.py'))
direct04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(direct04)
direct_v_travel, vec_to_angles = direct04.direct_v_travel, direct04.vec_to_angles
CENTERS, WIDTH, HIGH = direct04.CENTERS, direct04.WIDTH, direct04.HIGH

STEP_ORDER = [('bound', 'bound_alex'), ('messy', 'messy'), ('danny', 'danny'), ('nondh', 'nondh')]
LABELS = {'all': 'all stars', 'bound': '- bound (Alex)', 'messy': '- messy stream (Emily)',
          'danny': '- Danny streams', 'nondh': '- streams (Nondh)', 'rest': 'remaining'}
PLOT_LABELS = {'all': 'all stars', 'bound': 'Alex bound', 'messy': 'Emily messy streams',
               'danny': 'Danny streams', 'nondh': 'Nondh streams'}
PLOT_COLORS = {'all': 'cyan', 'bound': 'royalblue', 'messy': 'fuchsia', 'danny': 'tomato',
               'nondh': 'darkorange'}


def fmt(x):
    return f'{x:.4g}' if isinstance(x, (float, np.floating)) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', default='m12i')
    ap.add_argument('--snap', type=int, default=600)
    ap.add_argument('--weights', choices=['mass', 'number'], default='mass')
    ap.add_argument('--data-dir', default=os.environ.get('DIPOLE_DATA_DIR',
                                                         '/projectnb/fire-reflex/rpei/data'))
    args = ap.parse_args()
    tag = f'{args.sim}-{args.snap}'

    d = np.load(os.path.join(args.data_dir, f'{tag}-halostars.npz'))
    sm = np.load(os.path.join(args.data_dir, f'{tag}-streammasks.npz'))
    if not np.array_equal(d['star_index'], sm['star_index']):
        raise SystemExit('streammasks.npz is not aligned with halostars.npz: rerun 05')
    xyz = d['xyz'].astype(np.float64)
    v = d['v_xyz'].astype(np.float64)
    dist = d['dist'].astype(np.float64)
    w = d['mass'].astype(np.float64) if args.weights == 'mass' else np.ones(len(dist))
    keep = {'bound_alex': d['mask_bound_alex']} if 'mask_bound_alex' in d.files else {}
    keep.update({k[5:]: sm[k] for k in sm.files if k.startswith('mask_')})

    # cumulative keep-masks and the disjoint pieces each step removes
    steps = [('all', np.ones(len(dist), bool))]
    pieces = []
    cur = steps[0][1]
    for name, key in STEP_ORDER:
        if key not in keep:
            print(f'(no mask for {name}: step skipped)')
            continue
        pieces.append((name, cur & ~keep[key]))
        cur = cur & keep[key]
        steps.append((name, cur))
    pieces.append(('rest', cur))
    print('steps:', ' -> '.join(LABELS[s] for s, _ in steps))
    for name, m in pieces:
        print(f'  piece {name:6s}: {m.sum():9,d} stars, {100 * w[m].sum() / w.sum():5.2f}% of the '
              f'{args.weights} beyond {dist.min():.0f} kpc')

    def selection(analysis, c):
        if analysis == 'binned':
            return (dist > c - WIDTH / 2) & (dist < c + WIDTH / 2)
        return (dist > c) & (dist < HIGH)

    step_rows, comp_rows = [], []
    for analysis in ['binned', 'complementary']:
        for c in CENTERS:
            sel = selection(analysis, c)
            W_all = w[sel].sum()
            for name, m in steps:
                s = sel & m
                if s.sum() < 2:
                    step_rows.append([name, analysis, int(c), int(s.sum())] + [np.nan] * 8)
                    continue
                r = direct_v_travel(v[s], w[s], xyz[s])
                vt, th, ph = vec_to_angles(r['vec'])
                step_rows.append([name, analysis, int(c), r['n'], w[s].sum() / W_all, vt, th, ph,
                                  r['floor'], *r['vec']])
            # pieces: c_k = -(sum_{i in k} w_i v_i) / W_all ; sum_k c_k = v_travel(all)
            v_all = -(w[sel, None] * v[sel]).sum(axis=0) / W_all
            vhat = v_all / np.linalg.norm(v_all)
            for name, m in pieces:
                s = sel & m
                ck = -(w[s, None] * v[s]).sum(axis=0) / W_all
                comp_rows.append([name, analysis, int(c), int(s.sum()), w[s].sum() / W_all,
                                  float(ck @ vhat), float(np.linalg.norm(ck)), *ck])
            last = step_rows[-1]
            print(f'{analysis:13s} {c:3d} kpc  |v| all {np.linalg.norm(v_all):6.1f}  '
                  f'final step ({steps[-1][0]}) {last[5]:6.1f} +- {last[8]:4.1f} km/s', flush=True)

    suffix = '' if args.weights == 'mass' else '-number'
    f1 = os.path.join(args.data_dir, f'{tag}-progressive-direct{suffix}.csv')
    with open(f1, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['step', 'selection', 'center_kpc', 'N', 'weight_frac_kept', 'v_travel',
                     'theta_deg', 'phi_deg', 'shot_noise_floor', 'vx', 'vy', 'vz'])
        wr.writerows([[fmt(x) for x in r] for r in step_rows])
    f2 = os.path.join(args.data_dir, f'{tag}-progressive-components{suffix}.csv')
    with open(f2, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['piece', 'selection', 'center_kpc', 'N', 'weight_frac', 'c_along_vall',
                     'c_norm', 'cx', 'cy', 'cz'])
        wr.writerows([[fmt(x) for x in r] for r in comp_rows])
    print('saved', f1, '\nsaved', f2)

    # ---------------------------------------------------------------- plot --
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, analysis in zip(axes, ['binned', 'complementary']):
        for name, _ in steps:
            rr = np.array([r[2:] for r in step_rows if r[0] == name and r[1] == analysis], float)
            ax.plot(rr[:, 0], rr[:, 3], marker='o', ms=2.5, lw=1.5, color=PLOT_COLORS[name],
                    label=PLOT_LABELS[name])
        ax.set_title(analysis)
        ax.set_xlabel(r'$r$ [kpc]')
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r'$V_{\rm travel}$ [km/s]')
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f'{args.sim} progressive', fontsize=12)
    fig.tight_layout()
    out = os.path.join(args.data_dir, f'{tag}-progressive-direct{suffix}.png')
    fig.savefig(out, dpi=150)
    print('saved', out)


if __name__ == '__main__':
    main()
