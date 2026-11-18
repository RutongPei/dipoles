"""Dark matter vs stars: direct v_travel in m12i snapshot 600 after all masks.

Is the residual stellar v_travel (all masks applied, ~15-19 km/s complementary at 140-250 kpc)
the whole outer halo moving relative to the inner galaxy, or leftover stellar substructure?
Dark matter outweighs the stars ~285:1 at 20-305 kpc in m12i and has no stream catalogues, so it
shows what the halo as a whole does.

Stars: m12i-600-halostars.npz (02) + m12i-600-streammasks.npz (05), all masks applied in 06's
order: bound (Alex) -> messy -> Nondh -> Danny.

Dark matter: high-resolution DM ('dark', not 'dark2') read from the snapshot together with the
stars, so gizmo centres and rotates on the stars exactly as for the saved halo stars (checked
against the npz). Frame identical to the stars: v = (v - v_host) + H r, host principal axes.
Subhalo-bound DM is removed with Alex's bound-star criteria applied to DM:
  - halos: lowres.mass.frac < 1e-6, mass.bound/mass >= 0.4 (halo_analysis default),
    1e8 <= mass < host mass, vel.circ.max > 1 km/s (the same halo set as Alex's star assignment;
    for DM all of them count, including halos without stars)
  - members: within min(R_halo, 400 kpc) of the halo centre and |v - v_halo| (incl. Hubble flow)
    < 2 max(vel.std, vel.circ.max). Alex's iterative refinement is off
    (particle_number_fraction_converge = inf), so this is his full criterion.
This re-implements halo_analysis.assign_particle_indices with a KD-tree (the original loops over
all 70M DM particles per halo, ~45 min). Check: the same function is run on the saved halo stars
and compared with mask_bound_alex.

Direct estimator (same as 04/06): v_travel = -(sum m v)/(sum m) over the selection.
Selections: binned (5 kpc shells, centres 30-300) and complementary (c < r < 305 kpc).

Run on a compute node (reads 70M DM particles; ~10-15 GB; ~10-15 min, most of it in the DM
complementary selections of up to 25M particles; login nodes kill it at 15 min CPU):
    python scc/09_dm_vs_stars_m12i.py
    python scc/09_dm_vs_stars_m12i.py --show-dm-all    # also plot DM without the subhalo mask
    python scc/09_dm_vs_stars_m12i.py --replot         # plot from the csv only, no snapshot read

Outputs ($DIPOLE_RESULTS_DIR/m12i-600/, default /projectnb/fire-reflex/rpei/results/m12i-600/):
    m12i-600-dm-vs-stars.csv   rows: tracer (stars_all_masks, dm_bound_removed, dm_all) x selection x centre
    m12i-600-dm-vs-stars.png
"""
import argparse
import csv
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'code'))
_spec = importlib.util.spec_from_file_location('direct04', os.path.join(HERE, '04_direct_m12i.py'))
direct04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(direct04)
direct_v_travel, vec_to_angles = direct04.direct_v_travel, direct04.vec_to_angles
CENTERS, WIDTH, HIGH = direct04.CENTERS, direct04.WIDTH, direct04.HIGH

STAR_MASK_ORDER = ['bound_alex', 'messy', 'nondh', 'danny']    # 06's final step = all masks
R_MIN = 20.                                                    # same inner cut as the halo stars
HUBBLE = 0.1                                                   # km/s/kpc per unit h (100 km/s/Mpc)
TRACERS = {'stars_all_masks': dict(color='tomato', label='stars (all masks)'),
           'dm_bound_removed': dict(color='black', label='dark matter (subhalos removed)'),
           'dm_all': dict(color='0.6', ls='--', label='dark matter (all)')}

t0 = time.time()


def say(*a):
    print(f'[{time.time() - t0:7.1f} s]', *a, flush=True)


# --------------------------------------------------------------- subhalo mask --
def select_subhalos(hal, host_mass, lowres_max=1e-6, bound_frac_min=None, mass_min=1e8,
                    vmax_min=1.):
    """Halo indices with Alex's assign_particles_to_halo limits (the halo_analysis selection)."""
    if bound_frac_min is None:
        try:
            from halo_analysis import halo_io
            bound_frac_min = halo_io.BOUND_MASS_FRAC_MIN
        except Exception:
            bound_frac_min = 0.4
    ok = (hal.prop('lowres.mass.frac') < lowres_max)
    ok &= hal.prop('mass.bound/mass') >= bound_frac_min
    ok &= (hal['mass'] >= mass_min) & (hal['mass'] < host_mass)
    ok &= hal['vel.circ.max'] >= vmax_min
    return np.where(ok)[0]


def halos_in_host_frame(hal, idx, host_pos, host_vel, rotation, scalefactor, hubble):
    """Subhalo positions/velocities in the same frame as host.distance/velocity.principal."""
    import utilities.coordinate as uc
    d = (hal['position'][idx] - host_pos) * scalefactor            # kpc physical (zoom: no wrap)
    v = hal['velocity'][idx] - host_vel
    d_rot = uc.get_coordinates_rotated(d, rotation)
    v_rot = uc.get_coordinates_rotated(v, rotation) + hubble * d_rot
    return d_rot, v_rot


def bound_to_subhalos(xyz, v, halo_xyz, halo_v, halo_radius, halo_vel, radius_max=400.,
                      radius_factor=1., velocity_factor=2., hubble=0.0702, tree=None):
    """True for particles within Alex's phase-space limits of any of the given subhalos.

    xyz, v: particles in the host frame (Hubble flow included in v); halo_*: same frame.
    halo_vel = max(vel.std, vel.circ.max). The Hubble term is already in both velocities, so
    v - v_halo is the peculiar difference plus H (r - r_halo), as in halo_analysis.
    """
    from scipy.spatial import cKDTree
    if tree is None:
        tree = cKDTree(xyz)
    bound = np.zeros(len(xyz), bool)
    for hx, hv, hr, hvel in zip(halo_xyz, halo_v, halo_radius, halo_vel):
        rmax = min(radius_factor * hr, radius_max)
        cand = np.asarray(tree.query_ball_point(hx, rmax), dtype=np.int64)
        if cand.size == 0:
            continue
        dv = np.linalg.norm(v[cand] - hv, axis=1)
        bound[cand[dv < velocity_factor * hvel]] = True
    return bound


# ---------------------------------------------------------------- estimators --
def selection(dist, analysis, c):
    if analysis == 'binned':
        return (dist > c - WIDTH / 2) & (dist < c + WIDTH / 2)
    return (dist > c) & (dist < HIGH)


def direct_rows(name, xyz, v, w, dist, keep):
    rows = []
    say(f'direct v_travel for {name} ({keep.sum():,} particles, 2 x {len(CENTERS)} selections)')
    for analysis in ['binned', 'complementary']:
        for c in CENTERS:
            s = keep & selection(dist, analysis, c)
            if s.sum() < 2:
                rows.append([name, analysis, int(c), int(s.sum())] + [np.nan] * 8)
                continue
            r = direct_v_travel(v[s], w[s], xyz[s])
            vt, th, ph = vec_to_angles(r['vec'])
            rows.append([name, analysis, int(c), r['n'], vt, th, ph, r['floor'], *r['vec'],
                         r['lopsided']])
        say(f'  {name} {analysis} done')
    return rows


HEADER = ['tracer', 'selection', 'center_kpc', 'N', 'v_travel', 'theta_deg', 'phi_deg',
          'shot_noise_floor', 'vx', 'vy', 'vz', 'lopsided']


# ---------------------------------------------------------------------- plot --
def plot(csv_path, out_png, sim, tracers):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, analysis in zip(axes, ['binned', 'complementary']):
        for t in tracers:
            rr = [r for r in rows if r['tracer'] == t and r['selection'] == analysis]
            if not rr:
                continue
            x = np.array([float(r['center_kpc']) for r in rr])
            y = np.array([float(r['v_travel']) for r in rr])
            st = TRACERS[t]
            ax.plot(x, y, marker='o', ms=2.5, lw=1.5, color=st['color'], ls=st.get('ls', '-'),
                    label=st['label'])
        ax.set_title(analysis)
        ax.set_xlabel(r'$r$ [kpc]')
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r'$V_{\rm travel}$ [km/s]')
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f'{sim}: dark matter vs stars', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    say('saved', out_png)


# ---------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', default='m12i')
    ap.add_argument('--snap', type=int, default=600)
    ap.add_argument('--show-dm-all', action='store_true', help='also plot DM without subhalo mask')
    ap.add_argument('--replot', action='store_true', help='only redraw the figure from the csv')
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
    out_csv = os.path.join(out_dir, f'{tag}-dm-vs-stars.csv')
    out_png = os.path.join(out_dir, f'{tag}-dm-vs-stars.png')
    tracers = ['stars_all_masks', 'dm_bound_removed'] + (['dm_all'] if args.show_dm_all else [])
    if args.replot:
        plot(out_csv, out_png, args.sim, tracers)
        return

    # ---- stars: saved halo stars with all masks ----
    d = np.load(os.path.join(args.data_dir, f'{tag}-halostars.npz'))
    sm = np.load(os.path.join(args.data_dir, f'{tag}-streammasks.npz'))
    if not np.array_equal(d['star_index'], sm['star_index']):
        raise SystemExit('streammasks.npz is not aligned with halostars.npz: rerun 05')
    s_xyz = d['xyz'].astype(np.float64)
    s_v = d['v_xyz'].astype(np.float64)
    s_dist = d['dist'].astype(np.float64)
    s_m = d['mass'].astype(np.float64)
    keep_masks = {'bound_alex': d['mask_bound_alex']}
    keep_masks.update({k[5:]: sm[k] for k in sm.files if k.startswith('mask_')})
    s_keep = np.ones(len(s_dist), bool)
    for k in STAR_MASK_ORDER:
        if k not in keep_masks:
            raise SystemExit(f'missing star mask {k}; available: {sorted(keep_masks)}')
        s_keep &= keep_masks[k]
    say(f'stars r > {R_MIN:.0f} kpc: {len(s_dist):,}; after all masks {s_keep.sum():,} '
        f'({100 * s_m[s_keep].sum() / s_m.sum():.1f}% of the mass)')

    # ---- snapshot: stars (for the host frame) + high-res DM ----
    import gizmo_analysis as gizmo
    import dipole
    sim_dir = dipole.get_simulation_directory(args.sim)
    say('reading stars + dark matter from', sim_dir)
    part = gizmo.io.Read.read_snapshots(['star', 'dark'], 'index', args.snap, sim_dir,
                                        properties=['position', 'velocity', 'mass',
                                                    'form.scalefactor'],
                                        assign_hosts_rotation=True)
    hal = dipole.load_halo_catalog(args.sim, args.snap)
    host_pos, host_vel = part.host['position'][0], part.host['velocity'][0]
    rot = part.host['rotation'][0]
    a = part.snapshot['scalefactor']
    z = part.snapshot['redshift']
    hubble = HUBBLE * part.info['hubble'] * np.sqrt(part.info['omega_matter'] * (1 + z)**3 +
                                                    part.info['omega_lambda'])   # H(z) [km/s/kpc]
    say(f'host position {np.round(host_pos, 2)}, velocity {np.round(host_vel, 2)}, H = {hubble:.4f} km/s/kpc')
    for k, ref in [('host_position', host_pos), ('host_velocity', host_vel), ('host_rotation', rot)]:
        if k in d.files:
            say(f'  {k}: max |difference| from the saved halo stars = {np.abs(d[k] - ref).max():.2e}')

    # sanity: reproduce the saved star frame for a few saved stars
    test = d['star_index'][:2000]
    dx = np.abs(part['star'].prop('host.distance.principal', test) - d['xyz'][:2000]).max()
    dv = np.abs(part['star'].prop('host.velocity.principal', test) - d['v_xyz'][:2000]).max()
    say(f'  saved-star frame check: max |dx| = {dx:.2e} kpc, max |dv| = {dv:.2e} km/s')

    # ---- subhalo selection, in the host frame ----
    host_index = np.argmin(np.sum(hal['host.distance']**2, axis=1))
    sub = select_subhalos(hal, hal['mass'][host_index])
    sub = sub[sub != host_index]
    h_xyz, h_v = halos_in_host_frame(hal, sub, host_pos, host_vel, rot, a, hubble)
    h_r = hal['radius'][sub]
    h_vel = np.maximum(hal['vel.std'][sub], hal['vel.circ.max'][sub])
    say(f'{len(sub)} subhalos within Alex\'s limits (>= 1e8 Msun); '
        f'{np.sum(np.linalg.norm(h_xyz, axis=1) < HIGH + 50)} within {HIGH + 50:.0f} kpc of the host')

    # ---- check the re-implementation on the saved stars ----
    s_bound = bound_to_subhalos(s_xyz, s_v, h_xyz, h_v, h_r, h_vel, hubble=hubble)
    alex = ~keep_masks['bound_alex']
    both = np.sum(s_bound & alex)
    say(f'check on stars (r > {R_MIN:.0f}): this cut flags {s_bound.sum():,}, Alex\'s mask '
        f'{alex.sum():,}, both {both:,} ({100 * both / max(alex.sum(), 1):.2f}% of Alex\'s)')

    # ---- dark matter in 20-305 kpc ----
    dm_r = part['dark'].prop('host.distance.total')
    idx = np.where((dm_r > R_MIN) & (dm_r < HIGH))[0]
    del dm_r
    dm_xyz = part['dark'].prop('host.distance.principal', idx).astype(np.float64)
    dm_v = part['dark'].prop('host.velocity.principal', idx).astype(np.float64)
    dm_m = part['dark']['mass'][idx].astype(np.float64)
    dm_dist = np.linalg.norm(dm_xyz, axis=1)
    say(f'dark matter {R_MIN:.0f}-{HIGH:.0f} kpc: {len(idx):,} particles '
        f'(of {len(part["dark"]["mass"]):,}), particle mass {np.median(dm_m):.3g} Msun; '
        f'DM/star mass there = {dm_m.sum() / s_m[s_dist < HIGH].sum():.1f}')
    del part
    dm_bound = bound_to_subhalos(dm_xyz, dm_v, h_xyz, h_v, h_r, h_vel, hubble=hubble)
    say(f'subhalo-bound DM: {dm_bound.sum():,} particles ({100 * dm_m[dm_bound].sum() / dm_m.sum():.2f}% of the mass)')

    # ---- direct v_travel ----
    rows = direct_rows('stars_all_masks', s_xyz, s_v, s_m, s_dist, s_keep)
    rows += direct_rows('dm_bound_removed', dm_xyz, dm_v, dm_m, dm_dist, ~dm_bound)
    rows += direct_rows('dm_all', dm_xyz, dm_v, dm_m, dm_dist, np.ones(len(dm_dist), bool))
    with open(out_csv, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(HEADER)
        wr.writerows([[f'{x:.4g}' if isinstance(x, (float, np.floating)) else x for x in r]
                      for r in rows])
    say('saved', out_csv)

    # summary: complementary, and the angle between stars and DM vectors
    print(f'\n{"selection":13s} {"c":>4s} {"N stars":>9s} {"V stars":>8s} {"N DM":>10s} '
          f'{"V DM":>7s} {"V DM all":>9s} {"angle(stars, DM)":>17s}')
    by = {(r[0], r[1], r[2]): r for r in rows}
    for analysis in ['binned', 'complementary']:
        for c in CENTERS[::2]:
            s, m, m0 = (by[(t, analysis, int(c))] for t in
                        ['stars_all_masks', 'dm_bound_removed', 'dm_all'])
            vs, vd = np.array(s[8:11], float), np.array(m[8:11], float)
            ang = np.degrees(np.arccos(np.clip(vs @ vd / (np.linalg.norm(vs) * np.linalg.norm(vd)),
                                               -1, 1)))
            print(f'{analysis:13s} {c:4d} {s[3]:9,d} {s[4]:8.1f} {m[3]:10,d} {m[4]:7.1f} '
                  f'{m0[4]:9.1f} {ang:13.0f} deg')

    plot(out_csv, out_png, args.sim, tracers)


if __name__ == '__main__':
    main()
