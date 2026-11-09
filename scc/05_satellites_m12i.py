"""Satellite census for m12i snapshot 600: which bound satellites Alex's mask removes,
their phase-space properties (for fake-satellite injection), and how much each one
contributes to the all-stars direct v_travel.

No snapshot read. Needs, from 02_read_m12i_600.py --alex-bound:
    $DIPOLE_DATA_DIR/m12i-600-halostars.npz
    $DIPOLE_DATA_DIR/particle-assignments/m12i-600-bound-star-indices.npy  (Alex's per-halo indices)
and the Rockstar catalogue on SCC (read with dipole.load_halo_catalog, ~seconds).

    python scc/05_satellites_m12i.py              # satellites with >= 50 stars at r > 20 kpc
    python scc/05_satellites_m12i.py --nmin 200

Output: $DIPOLE_DATA_DIR/m12i-600-satellites.csv, one row per satellite, all vectors in the
host principal frame (same frame as xyz, v_xyz in the npz; z = disc normal):
    halo_index, N (r > 20 kpc), mstar, distance, x, y, z (mass-weighted COM), lon, lat (deg, as in the
    velocity maps), vx, vy, vz (COM velocity, host frame), v_r, v_theta, v_phi (at the COM),
    sigma_1d (internal), r50, r90 (kpc from COM), halo mass, vmax, R_halo, bound fraction
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
import dipole  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--sim', default='m12i')
ap.add_argument('--snap', type=int, default=600)
ap.add_argument('--nmin', type=int, default=50, help='min member stars at r > 20 kpc to list')
ap.add_argument('--centres', default='120,150,160,170,200,230,240',
                help='complementary inner edges for the momentum breakdown [kpc]')
args = ap.parse_args()

D = dipole.get_data_directory()
tag = f'{args.sim}-{args.snap}'
d = np.load(os.path.join(D, f'{tag}-halostars.npz'))
xyz, v, m, dist = (d[k].astype(np.float64) for k in ['xyz', 'v_xyz', 'mass', 'dist'])
star_index = d['star_index']                       # sorted (np.where in 02)
alex = np.load(os.path.join(D, 'particle-assignments', f'{tag}-bound-star-indices.npy'),
               allow_pickle=True)

try:
    hal = dipole.load_halo_catalog(args.sim, args.snap)
except Exception as e:                             # census still works without the catalogue
    print(f'(halo catalogue not read: {type(e).__name__}: {e})')
    hal = None


def hprop(name, i):
    try:
        return float(hal.prop(name, i))
    except Exception:
        return np.nan


def to_rows(idx):
    """full-snapshot star indices -> rows of the npz (stars with r > 20 kpc)"""
    idx = np.asarray(idx, dtype=np.int64)
    pos = np.searchsorted(star_index, idx)
    pos = np.clip(pos, 0, len(star_index) - 1)
    return pos[star_index[pos] == idx]


def sph_at(vec, x):
    r = np.linalg.norm(x)
    th, ph = np.arccos(x[2] / r), np.arctan2(x[1], x[0])
    rh = x / r
    thh = np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)])
    phh = np.array([-np.sin(ph), np.cos(ph), 0.])
    return vec @ rh, vec @ thh, vec @ phh


# ------------------------------------------------------------ census --------
sats = []
for h, idx in enumerate(alex):
    if len(idx) == 0:
        continue
    rows = to_rows(idx)
    if len(rows) < args.nmin:
        continue
    w = m[rows]
    com = (w[:, None] * xyz[rows]).sum(0) / w.sum()
    vcom = (w[:, None] * v[rows]).sum(0) / w.sum()
    rr = np.linalg.norm(xyz[rows] - com, axis=1)
    sig = np.sqrt(np.mean(np.sum((v[rows] - vcom) ** 2, axis=1)) / 3)
    vr, vt, vp = sph_at(vcom, com)
    sats.append(dict(
        halo_index=h, N=len(rows), mstar=w.sum(), distance=np.linalg.norm(com),
        x=com[0], y=com[1], z=com[2],
        lon=np.degrees(np.arctan2(com[1], com[0])), lat=np.degrees(np.arcsin(com[2] / np.linalg.norm(com))),
        vx=vcom[0], vy=vcom[1], vz=vcom[2], v_r=vr, v_theta=vt, v_phi=vp,
        sigma_1d=sig, r50=np.percentile(rr, 50), r90=np.percentile(rr, 90),
        halo_mass=hprop('mass', h), vmax=hprop('vel.circ.max', h), R_halo=hprop('radius', h),
        bound_frac=hprop('mass.bound/mass', h), rows=rows))
sats.sort(key=lambda s: -s['N'])

print(f'\n=== satellites in Alex\'s assignment with >= {args.nmin} stars at r > 20 kpc ===')
print(f"{'halo':>6} {'N':>7} {'M*':>9} {'d':>6} {'lon':>6} {'lat':>6} {'|V|':>6} "
      f"{'v_r':>6} {'v_th':>6} {'v_ph':>6} {'sig':>5} {'r50':>5} {'r90':>5} "
      f"{'Mhalo':>8} {'vmax':>5} {'Rhalo':>6} {'fbound':>6}")
for s in sats:
    print(f"{s['halo_index']:6d} {s['N']:7,d} {s['mstar']:9.2e} {s['distance']:6.1f} {s['lon']:6.1f} "
          f"{s['lat']:6.1f} {np.linalg.norm([s['vx'], s['vy'], s['vz']]):6.1f} {s['v_r']:6.1f} "
          f"{s['v_theta']:6.1f} {s['v_phi']:6.1f} {s['sigma_1d']:5.1f} {s['r50']:5.1f} {s['r90']:5.1f} "
          f"{s['halo_mass']:8.2e} {s['vmax']:5.1f} {s['R_halo']:6.1f} {s['bound_frac']:6.2f}")

cols = [k for k in sats[0] if k != 'rows'] if sats else []
out = os.path.join(D, f'{tag}-satellites.csv')
with open(out, 'w') as f:
    f.write(','.join(cols) + '\n')
    for s in sats:
        f.write(','.join(f'{s[k]:.6g}' if isinstance(s[k], float) else str(s[k]) for k in cols) + '\n')
print('saved', out)

# ------------------------------------- momentum breakdown of direct v_travel --
# all-stars direct (number-weighted, like the fit) = sum over satellites + rest:
#   -<v>_sel = sum_s ( -sum_{i in s, sel} v_i / N_sel ) + ( -sum_{i in rest, sel} v_i / N_sel )
print('\n=== all-stars complementary direct v_travel, split by satellite (number-weighted, km/s) ===')
top = sats[:5]
for c in [float(x) for x in args.centres.split(',')]:
    sel = (dist > c) & (dist < 305)
    n = sel.sum()
    total = -v[sel].sum(0) / n
    print(f'\ninner edge {c:.0f} kpc: N = {n:,}, total -<v> = {np.round(total, 1)}  |{np.linalg.norm(total):.1f}|')
    in_sat = np.zeros(len(dist), bool)
    for s in top:
        r = s['rows'][sel[s['rows']]]
        in_sat[r] = True
        if len(r):
            part = -v[r].sum(0) / n
            print(f"   halo {s['halo_index']:6d} at {s['distance']:5.0f} kpc: {len(r):6,d} stars "
                  f"({len(r) / n:5.1%})  contributes {np.round(part, 1)}  |{np.linalg.norm(part):.1f}|")
    rest = sel & ~in_sat
    part = -v[rest].sum(0) / n
    print(f"   everything else              : {rest.sum():6,d} stars          "
          f"contributes {np.round(part, 1)}  |{np.linalg.norm(part):.1f}|")

# ------------------------------------------------- mask cross-checks --------
if hal is not None and 'star.indices' in hal:
    print('\n=== Rockstar (star_600) vs Alex: halos with >= 50 Rockstar stars at 20-305 kpc ===')
    print('   (a satellite with Rockstar stars but none in Alex\'s assignment slipped through his cuts)')
    for h in range(len(hal['star.indices'])):
        ri = hal['star.indices'][h]
        if ri is None or len(ri) == 0:
            continue
        rows = to_rows(ri)
        rows = rows[dist[rows] < 305]
        if len(rows) < 50 or hprop('host.distance.total', h) == 0:
            continue
        na = len(to_rows(alex[h])) if h < len(alex) else 0
        flag = '  <-- NOT in Alex mask' if na == 0 else ''
        print(f'   halo {h:6d}: d = {np.median(dist[rows]):6.1f} kpc, Rockstar N = {len(rows):7,d}, '
              f'Alex N = {na:7,d}, Mhalo = {hprop("mass", h):.2e}, fbound = {hprop("mass.bound/mass", h):.2f}, '
              f'lowres = {hprop("lowres.mass.frac", h):.1e}{flag}')
