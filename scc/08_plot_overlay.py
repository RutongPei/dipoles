"""One-panel figure in the format of Alex's XMC talk (m12f-463-dipole-distance): binned and
complementary V_travel for one masking step, overlaid, with fixed axes so hosts/snapshots compare
directly. Reads the csv written by 06_progressive_direct_m12i.py; no snapshot needed.

Alex's figure labels the complementary curve 'cumulative'; here it is called complementary
(r > c, < 305 kpc), which is what both his and our code compute.

    python scc/08_plot_overlay.py --sim m12f --snap 463                 # bound removed
    python scc/08_plot_overlay.py --sim m12f --snap 463 --vlim 53       # + Alex's v_dipole^lim line
    python scc/08_plot_overlay.py --sim m12i --snap 600 --step danny    # any step from 06

Output: $DIPOLE_DATA_DIR/{sim}-{snap}-{step}-overlay.png
"""
import argparse
import csv
import os

import numpy as np

COLORS = {'binned': 'cyan', 'complementary': 'royalblue'}
STEP_TITLES = {'all': 'all stars', 'bound': 'bound removed', 'messy': 'bound + messy stream removed',
               'stream': 'bound + z0 stream table removed', 'nondh': 'up to Nondh streams removed',
               'danny': 'all masks'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', default='m12f')
    ap.add_argument('--snap', type=int, default=463)
    ap.add_argument('--step', default='bound', help='masking step from the 06 csv (default bound)')
    ap.add_argument('--weights', choices=['mass', 'number'], default='mass')
    ap.add_argument('--vlim', type=float, default=None,
                    help='draw a dashed reference line at this V [km/s] (Alex: 53 for m12f 463)')
    ap.add_argument('--xlim', type=float, nargs=2, default=[25, 305])
    ap.add_argument('--ylim', type=float, nargs=2, default=[0, 100])
    ap.add_argument('--data-dir', default=os.environ.get('DIPOLE_DATA_DIR',
                                                         '/projectnb/fire-reflex/rpei/data'),
                    help='inputs: halostars/streammasks npz, catalogues')
    ap.add_argument('--results-dir', default=os.environ.get('DIPOLE_RESULTS_DIR',
                                                            '/projectnb/fire-reflex/rpei/results'),
                    help='outputs go to <results-dir>/<sim>-<snap>/')
    args = ap.parse_args()
    tag = f'{args.sim}-{args.snap}'
    out_dir = os.path.join(args.results_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    suffix = '' if args.weights == 'mass' else '-number'

    with open(os.path.join(out_dir, f'{tag}-progressive-direct{suffix}.csv')) as f:
        rows = [r for r in csv.DictReader(f) if r['step'] == args.step]
    if not rows:
        raise SystemExit(f'no step {args.step!r} in the csv (run 06 first)')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for sel in ['binned', 'complementary']:
        rr = [r for r in rows if r['selection'] == sel]
        c = np.array([float(r['center_kpc']) for r in rr])
        v = np.array([float(r['v_travel']) for r in rr])
        ax.plot(c, v, lw=2, color=COLORS[sel], label=sel)
    if args.vlim is not None:
        ax.axhline(args.vlim, ls='--', color='0.4', lw=1.2,
                   label=rf'$v_{{\rm dipole}}^{{\rm lim}} \simeq {args.vlim:g}$ km/s')
    ax.set_xlim(*args.xlim)
    ax.set_ylim(*args.ylim)
    ax.set_xlabel(r'$r$ [kpc]')
    ax.set_ylabel(r'$V_{\rm travel}$ [km/s]')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    title = f'{args.sim} {STEP_TITLES.get(args.step, args.step)}'
    if args.snap != 600:
        title += f' (snapshot {args.snap})'
    ax.set_title(title)
    fig.tight_layout()
    out = os.path.join(out_dir, f'{tag}-{args.step}-overlay{suffix}.png')
    fig.savefig(out, dpi=150)
    print('saved', out)


if __name__ == '__main__':
    main()
