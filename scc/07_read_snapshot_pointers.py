"""Read any snapshot of a host (e.g. m12f 463, the LMC-analogue pericentre), save the halo stars
and the masks that 06_progressive_direct_m12i.py reads:

    {sim}-{snap}-halostars.npz    same format as 02_read_m12i_600.py (r > 20 kpc), incl. mask_bound_alex
    {sim}-{snap}-streammasks.npz  star_index [+ mask_stream, mask_danny, mask_danny_gtr1e9, mask_nondh]

Bound mask: Alex's particle-assignments/{sim}-{snap}-bound-star-indices.npy (indices into this
snapshot); if missing, made from the Rockstar catalogue with dipole.assign_particles_to_halo.

z = 0 lists (stream table, Danny, Nondh) are only used if they can be mapped to this snapshot:
  - snap 600: used directly;
  - other snapshots: needs <sim dir>/track/star_gas_pointers_{snap}.hdf5. Then
        ptr = part.Pointer.get_pointers(species_name_from='star', species_names_to='star')
    is indexed by the z = 0 star index and gives the index here; ptr[z0_inds], keep >= 0
    (0 is valid; negative = not a star at this snapshot). Without the file these masks are skipped.
The stream table (default latte/m12f_res7100/m12f_z0_stream.hdf5) is an astropy table; it is read
with h5py (fire_env has no astropy). Columns used: inds (z = 0 star index), t_strip.

Then:  python scc/06_progressive_direct_m12i.py --sim m12f --snap 463

Run from the repo top level (snapshot read: ~10-20 GB; fine on a login node if under 15 min CPU):
    python scc/07_read_snapshot_pointers.py --sim m12f --snap 463
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))

t0 = time.time()


def say(*a):
    print(f'[{time.time() - t0:7.1f} s]', *a, flush=True)


def load_flat(path):
    a = np.load(path, allow_pickle=True)
    if a.dtype == object:
        a = np.hstack([np.atleast_1d(x) for x in a if len(np.atleast_1d(x))])
    return np.unique(a.astype(np.int64))


def read_stream_table(path):
    import h5py
    with h5py.File(path, 'r') as f:
        t = f['__astropy_table__'][:]
    print(f'[stream table] {path}: {len(t):,} rows, columns {t.dtype.names}')
    return t


def sapfire_z0_indices(path, sim, col='st_index'):
    with open(path) as f:
        rd = csv.DictReader(f)
        cols = {c.strip().lower(): c for c in rd.fieldnames}
        rows = [r for r in rd if 'simname' not in cols or sim in r[cols['simname']]]
    return np.unique(np.array([int(float(r[cols[col]])) for r in rows], np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', default='m12f')
    ap.add_argument('--snap', type=int, default=463)
    ap.add_argument('--stream-table', default=None,
                    help='default: <sim dir>/<sim>_z0_stream.hdf5 if it exists')
    ap.add_argument('--nondh', default=None, help='SAPFIRE csv for this host (z = 0 st_index)')
    ap.add_argument('--rmin-save', type=float, default=20.)
    args = ap.parse_args()

    import dipole  # noqa: E402
    data_dir = dipole.get_data_directory()
    sim_dir = dipole.get_simulation_directory(args.sim)
    snap, tag = args.snap, f'{args.sim}-{args.snap}'

    pfile = os.path.join(sim_dir, 'track', f'star_gas_pointers_{snap:03d}.hdf5')
    use_pointers = snap != 600 and os.path.exists(pfile)
    can_map = snap == 600 or use_pointers
    if not can_map:
        print(f'no pointer file {pfile}: z = 0 lists (stream table, Danny, Nondh) are skipped, '
              'bound mask only')

    say(f'reading {args.sim} snapshot {snap} from {sim_dir}')
    part = dipole.gizmo.io.Read.read_snapshots(
        ['star'], 'index', snap, simulation_directory=sim_dir,
        assign_hosts_rotation=True, assign_pointers=use_pointers)
    star = part['star']
    n_here = len(star['mass'])
    say(f'{n_here:,} star particles; z = {part.snapshot["redshift"]:.4f}, '
        f't = {part.snapshot["time"]:.3f} Gyr')

    xyz_all = star.prop('host.distance.principal').astype(np.float64)
    vxyz_all = star.prop('host.velocity.principal').astype(np.float64)
    r_all = np.sqrt(np.sum(xyz_all**2, axis=1))
    inner = r_all < 15
    L = np.sum(star['mass'][inner, None] * np.cross(xyz_all[inner], vxyz_all[inner]), axis=0)
    print(f'host position {np.round(part.host["position"][0], 2)}, '
          f'cos(L(<15 kpc), principal z) = {L[2] / np.linalg.norm(L):+.4f} (want |.| ~ 1)')

    # ------------------------------------------------------------ bound ---------
    print('\n=== bound stars at this snapshot ===')
    pa = os.path.join(data_dir, 'particle-assignments')
    f_bound = os.path.join(pa, f'{tag}-bound-star-indices.npy')
    groups = None
    if os.path.exists(f_bound):
        groups = [np.asarray(x, np.int64) for x in np.load(f_bound, allow_pickle=True)]
        print(f"Alex's file {os.path.basename(f_bound)}: {len(groups):,} halos")
    else:
        try:
            say(f'no {os.path.basename(f_bound)}; reading Rockstar and assigning stars (slow)')
            hal = dipole.load_halo_catalog(args.sim, snap)
            hal = dipole.assign_particles_to_halo(sim=args.sim, snapshot=snap, part=part, halos=hal)
            groups = [np.asarray(x, np.int64) for x in hal['star.indices']]
        except Exception as ex:
            print(f'! no bound mask: {type(ex).__name__}: {ex}')
    keep_bound = np.ones(n_here, bool)
    bound_here = np.array([], np.int64)
    if groups is not None:
        allidx = np.hstack([g for g in groups if len(g)])
        if allidx.max() >= n_here:
            sys.exit(f'bound file has index {allidx.max():,} >= {n_here:,} stars: not this snapshot')
        order = np.argsort([len(g) for g in groups])[::-1][:6]
        print('largest groups (halo index: N, centre distance from host, median |x - centre|):')
        for i in order:
            x = xyz_all[groups[i]]
            cen = np.median(x, axis=0)
            d = np.linalg.norm(cen)
            spread = np.median(np.linalg.norm(x - cen, axis=1))
            flag = '  <- HOST? excluded' if d < 5 else ''
            print(f'  {i:7d}: {len(groups[i]):9,d}  {d:7.1f} kpc  {spread:6.2f} kpc{flag}')
            if d < 5:
                groups[i] = np.array([], np.int64)
        print("  (compact groups = this snapshot's star order matches the file)")
        bound_here = np.unique(np.hstack([g for g in groups if len(g)]))
        keep_bound[bound_here] = False
        sel20 = r_all > args.rmin_save
        print(f'bound stars: {len(bound_here):,}; {100 * star["mass"][~keep_bound & sel20].sum() / star["mass"][sel20].sum():.2f}% '
              f'of the stellar mass beyond {args.rmin_save:g} kpc')

    # --------------------------------------------- z = 0 lists (if mappable) ----
    here = {}
    if can_map:
        if use_pointers:
            ptr = part.Pointer.get_pointers(species_name_from='star', species_names_to='star')
            print(f'\npointers: {len(ptr):,} z = 0 stars, {np.sum(ptr >= 0):,} are stars here')
        else:
            ptr = np.arange(n_here)

        def to_here(z0, name):
            z0 = np.asarray(z0, np.int64)
            if z0.max() >= len(ptr):
                print(f'! {name}: index {z0.max():,} >= {len(ptr):,} z = 0 stars')
                z0 = z0[z0 < len(ptr)]
            h = ptr[z0]
            h = np.unique(h[h >= 0])
            print(f'[{name}] {len(z0):,} z = 0 stars -> {len(h):,} here; '
                  f'{np.isin(h, bound_here).mean():.3f} of them bound here')
            return h

        st_path = args.stream_table or os.path.join(sim_dir, f'{args.sim}_z0_stream.hdf5')
        if os.path.exists(st_path):
            t = read_stream_table(st_path)
            here['stream'] = to_here(t['inds'], 'stream table')
            if 't_strip' in t.dtype.names and use_pointers:
                early = t['inds'][t['t_strip'] < part.snapshot['time']]
                print(f'  t_strip < t(snap) = {part.snapshot["time"]:.2f} Gyr: {len(early):,} stars')
                if len(early):
                    here['stream_stripped'] = to_here(early, 'stream table, stripped before snap')
        ddir = os.path.join(data_dir, 'danny-streams')
        here['danny'] = to_here(load_flat(os.path.join(ddir, f'mergers-{args.sim}-all.npy')), 'danny all')
        g = os.path.join(ddir, f'gtr-10-9-mergers-ids-{args.sim}.npy')
        if os.path.exists(g):
            here['danny_gtr1e9'] = to_here(load_flat(g), 'danny gtr-1e9')
        if args.nondh:
            here['nondh'] = to_here(sapfire_z0_indices(args.nondh, args.sim), 'nondh')

    # ------------------------------------------------------------- save ---------
    sel = r_all > args.rmin_save
    v_sph = dipole.vel_xyz_to_sph_optimized(vxyz_all[sel], xyz_all[sel])
    out = os.path.join(data_dir, f'{tag}-halostars.npz')
    np.savez(out,
             xyz=xyz_all[sel].astype(np.float32), v_xyz=vxyz_all[sel].astype(np.float32),
             v_sph=v_sph.astype(np.float32), dist=r_all[sel].astype(np.float32),
             mass=star['mass'][sel].astype(np.float32),
             feh=star.prop('metallicity.iron')[sel].astype(np.float32),
             star_index=np.where(sel)[0],
             **({'mask_bound_alex': keep_bound[sel]} if groups is not None else {}),
             host_position=part.host['position'][0], host_velocity=part.host['velocity'][0],
             host_rotation=part.host['rotation'][0], host_axis_ratios=part.host['axis.ratios'][0],
             redshift=part.snapshot['redshift'], snapshot=snap)
    say(f'saved {sel.sum():,} stars with r > {args.rmin_save:g} kpc to {out}')

    masks = {}
    for k, inds in here.items():
        m = np.ones(n_here, bool)
        m[inds] = False
        masks[f'mask_{k}'] = m[sel]
    out2 = os.path.join(data_dir, f'{tag}-streammasks.npz')
    np.savez(out2, star_index=np.where(sel)[0], **masks)
    print('saved', out2, f'(masks: {list(masks) or "none"}; True = keep)')
    print(f'\nnext: python scc/06_progressive_direct_m12i.py --sim {args.sim} --snap {snap}')


if __name__ == '__main__':
    main()
