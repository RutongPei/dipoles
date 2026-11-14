"""Read a single-file snapshot (e.g. Emily's m12f_z0_stream.hdf5), map the z = 0 stream lists
onto it with gizmo pointers, cross-check the star indexing against Danny's and Alex's files,
and save the two npz files that 06_progressive_direct_m12i.py reads:

    {sim}-{snap}-halostars.npz    same format as 02_read_m12i_600.py (r > 20 kpc stars)
    {sim}-{snap}-streammasks.npz  star_index + mask_danny [+ mask_nondh, mask_danny_gtr1e9]

Then the progressive plot is
    python scc/06_progressive_direct_m12i.py --sim m12f --snap <snap>

Pointers (Emily's recipe, trimmed). With assign_pointers=True,
    ptr = part.Pointer.get_pointers(species_name_from='star', species_names_to='star')
is indexed by the z = 0 star index and gives that star's index at this snapshot, so
    inds_here = ptr[z0_inds]; inds_here = inds_here[inds_here >= 0]
(>= 0, not > 0: 0 is a valid index; stars that were gas or did not exist yet are negative).
The forward pointers (forward=True, snapshot -> z = 0) are not needed for this.
At snapshot 600 the pointers are the identity and are not read.

gizmo only finds files called snapshot_NNN.hdf5 inside <sim dir>/output/, so the file is read
through a shadow simulation directory ($DIPOLE_DATA_DIR/sim-links/<sim>_res7100/) that symlinks
everything in the real one (snapshot_times.txt, track/, halo/, ...) plus
output/snapshot_NNN.hdf5 -> the file.

Checks printed:
  1. file structure, redshift -> snapshot index (snapshot_times.txt)
  2. pointer file present; z = 0 star count in the pointers vs the max index in Danny's lists
  3. Danny's lists mapped to this snapshot: fraction that exist here, radial distribution
  4. Alex's bound file for this snapshot (particle-assignments/{sim}-{snap}-bound-star-indices.npy):
     largest groups should be compact around their centre if the star order matches this file
  5. Alex's z = 0 bound stars mapped back with the pointers: the fraction still bound here
     (satellites at z = 0 were mostly satellites earlier too; ~0 would mean wrong pointers)

Run on a compute node (snapshot read needs ~20-30 GB), from the repo top level:
    python scc/07_read_snapshot_pointers.py --inspect-only        # 1 only, no snapshot read
    python scc/07_read_snapshot_pointers.py                       # m12f, Emily's file
    python scc/07_read_snapshot_pointers.py --nondh /projectnb/fire-reflex/rpei/data/sapfire/m12f.csv
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


# ---------------------------------------------------------------- 1. inspect --
def inspect_file(path):
    import h5py
    info = {'is_snapshot': False, 'redshift': None, 'n_star': None}
    with h5py.File(path, 'r') as f:
        print(f'=== {path} ({os.path.getsize(path) / 1e9:.2f} GB) ===')
        print('top-level:', ', '.join(f'{k} ({type(f[k]).__name__})' for k in f.keys()))
        if 'Header' in f:
            h = dict(f['Header'].attrs)
            for k in ('Redshift', 'Time', 'NumPart_Total', 'NumPart_ThisFile', 'NumFilesPerSnapshot',
                      'BoxSize', 'HubbleParam'):
                if k in h:
                    print(f'  Header.{k} = {h[k]}')
            info['redshift'] = float(h.get('Redshift', np.nan))
            if 'NumPart_Total' in h:
                info['n_star'] = int(h['NumPart_Total'][4])
            if int(h.get('NumFilesPerSnapshot', 1)) > 1:
                print('  ! NumFilesPerSnapshot > 1: this is one block of a multi-file snapshot')
        if 'PartType4' in f:
            g = f['PartType4']
            print('  PartType4:', ', '.join(f'{k}{g[k].shape}' for k in list(g.keys())[:20]))
            info['is_snapshot'] = 'Header' in f and 'Coordinates' in g
        if not info['is_snapshot']:
            print('\nnot a GIZMO snapshot layout; full contents:')
            f.visititems(lambda n, o: print(f'  {n}', getattr(o, 'shape', ''), getattr(o, 'dtype', ''))
                         if len(n.split('/')) <= 3 else None)
    return info


def snapshot_index_from_redshift(sim_dir, z):
    import utilities as ut
    Snapshot = ut.simulation.read_snapshot_times(sim_dir, verbose=False, error_if_no_file=False)
    if Snapshot is None or z is None or not np.isfinite(z):
        return None, None
    zs = np.asarray(Snapshot['redshift'])
    idx = np.asarray(Snapshot['index']) if 'index' in Snapshot else np.arange(len(zs))
    i = int(np.argmin(np.abs(zs - z)))
    return int(idx[i]), float(zs[i])


# ----------------------------------------------------------- shadow sim dir --
def make_shadow_dir(sim_dir, file_path, snap, data_dir, sim):
    shadow = os.path.join(data_dir, 'sim-links', f'{sim}_res7100')
    os.makedirs(os.path.join(shadow, 'output'), exist_ok=True)
    for name in os.listdir(sim_dir):
        if name == 'output':
            continue
        link = os.path.join(shadow, name)
        if not os.path.lexists(link):
            os.symlink(os.path.join(sim_dir, name), link)
    link = os.path.join(shadow, 'output', f'snapshot_{snap:03d}.hdf5')
    if os.path.lexists(link) and os.path.realpath(link) != os.path.realpath(file_path):
        sys.exit(f'{link} already points to {os.path.realpath(link)}; remove it first')
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(file_path), link)
    return shadow + '/'


# ------------------------------------------------------------------ lists ----
def load_flat(path):
    a = np.load(path, allow_pickle=True)
    if a.dtype == object:
        a = np.hstack([np.atleast_1d(x) for x in a if len(np.atleast_1d(x))])
    return np.unique(a.astype(np.int64))


def load_groups(path):
    """Alex's per-halo star index lists (object array)."""
    return [np.asarray(x, np.int64) for x in np.load(path, allow_pickle=True)]


def sapfire_z0_indices(path, sim, col='st_index'):
    with open(path) as f:
        rd = csv.DictReader(f)
        cols = {c.strip().lower(): c for c in rd.fieldnames}
        if col not in cols:
            sys.exit(f'{path}: no column {col!r}; have {list(cols)}')
        rows = [r for r in rd if 'simname' not in cols or sim in r[cols['simname']]]
    a = np.array([int(float(r[cols[col]])) for r in rows], np.int64)
    print(f'[nondh] {path}: {len(rows):,} rows for {sim}, {len(np.unique(a)):,} unique {col}')
    return np.unique(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', default='m12f')
    ap.add_argument('--file', default='/projectnb/fire-reflex/latte/m12f_res7100/m12f_z0_stream.hdf5')
    ap.add_argument('--snap', type=int, default=None,
                    help='snapshot index of the file (default: from its redshift)')
    ap.add_argument('--nondh', default=None, help='SAPFIRE csv for this host (z = 0 st_index)')
    ap.add_argument('--rmin-save', type=float, default=20.)
    ap.add_argument('--inspect-only', action='store_true')
    args = ap.parse_args()

    import dipole  # noqa: E402  (after argparse so --help is fast)
    data_dir = dipole.get_data_directory()
    sim_dir = dipole.get_simulation_directory(args.sim)

    # 1. file + snapshot index
    info = inspect_file(args.file)
    snap_z, z_tab = snapshot_index_from_redshift(sim_dir, info['redshift'])
    if snap_z is not None:
        print(f'redshift {info["redshift"]:.5f} -> snapshot {snap_z} (table z = {z_tab:.5f})')
    snap = args.snap if args.snap is not None else snap_z
    if args.snap is not None and snap_z is not None and args.snap != snap_z:
        print(f'! --snap {args.snap} differs from the redshift match ({snap_z})')
    if args.inspect_only:
        return
    if not info['is_snapshot']:
        sys.exit('file is not a GIZMO snapshot (see contents above); not reading it')
    if snap is None:
        sys.exit('cannot tell the snapshot index (no snapshot_times.txt?): pass --snap')
    use_pointers = snap != 600
    tag = f'{args.sim}-{snap}'

    if use_pointers:
        pfile = os.path.join(sim_dir, 'track', f'star_gas_pointers_{snap:03d}.hdf5')
        print(f'\npointer file {pfile}: {"found" if os.path.exists(pfile) else "MISSING"}')
        if not os.path.exists(pfile):
            tdir = os.path.join(sim_dir, 'track')
            have = sorted(os.listdir(tdir)) if os.path.isdir(tdir) else []
            print(f'  track/ has {len(have)} files, e.g. {have[:5]} ... {have[-5:]}')
            sys.exit('no pointers for this snapshot: Danny\'s z = 0 lists cannot be mapped')

    # 2. read
    shadow = make_shadow_dir(sim_dir, args.file, snap, data_dir, args.sim)
    say(f'reading {args.sim} snapshot {snap} from {shadow}output/ (-> {args.file})')
    part = dipole.gizmo.io.Read.read_snapshots(
        ['star'], 'index', snap, simulation_directory=shadow,
        assign_hosts_rotation=True, assign_pointers=use_pointers)
    star = part['star']
    n_here = len(star['mass'])
    say(f'{n_here:,} star particles; z = {part.snapshot["redshift"]:.4f}, '
        f't = {part.snapshot["time"]:.3f} Gyr')
    if info['n_star'] is not None and info['n_star'] != n_here:
        print(f'! header says {info["n_star"]:,} stars, read {n_here:,}')

    if use_pointers:
        ptr = part.Pointer.get_pointers(species_name_from='star', species_names_to='star')
        n_z0 = len(ptr)
        print(f'\npointers: {n_z0:,} z = 0 stars; {np.sum(ptr >= 0):,} of them are stars here; '
              f'max pointer {ptr.max():,} (< {n_here:,} required)')
        assert ptr.max() < n_here
    else:
        ptr = np.arange(n_here)
        n_z0 = n_here

    def to_here(z0_inds, name):
        z0_inds = np.asarray(z0_inds, np.int64)
        bad = z0_inds >= n_z0
        if bad.any():
            print(f'! {name}: {bad.sum():,} indices >= number of z = 0 stars ({n_z0:,}): wrong host/list?')
        h = ptr[z0_inds[~bad]]
        h = np.unique(h[h >= 0])
        print(f'[{name}] {len(z0_inds):,} z = 0 stars -> {len(h):,} stars here '
              f'({100 * len(h) / max(len(z0_inds), 1):.1f}%)')
        return h

    # host frame (same as 02)
    xyz_all = star.prop('host.distance.principal').astype(np.float64)
    vxyz_all = star.prop('host.velocity.principal').astype(np.float64)
    r_all = np.sqrt(np.sum(xyz_all**2, axis=1))
    inner = r_all < 15
    L = np.sum(star['mass'][inner, None] * np.cross(xyz_all[inner], vxyz_all[inner]), axis=0)
    print(f'\nhost position {np.round(part.host["position"][0], 2)}, '
          f'cos(L(<15 kpc), principal z) = {L[2] / np.linalg.norm(L):+.4f}')

    # 3. Danny's lists
    print('\n=== Danny streams (z = 0 indices) mapped here ===')
    ddir = os.path.join(data_dir, 'danny-streams')
    danny_here = to_here(load_flat(os.path.join(ddir, f'mergers-{args.sim}-all.npy')), 'danny all')
    gtr_path = os.path.join(ddir, f'gtr-10-9-mergers-ids-{args.sim}.npy')
    gtr_here = to_here(load_flat(gtr_path), 'danny gtr-1e9') if os.path.exists(gtr_path) else None
    if gtr_here is not None:
        print(f'  gtr-1e9 subset of all, here: {np.isin(gtr_here, danny_here).mean():.4f} (want 1)')
    h, e = np.histogram(r_all[danny_here], bins=[0, 20, 50, 100, 150, 200, 250, 300, 400, np.inf])
    print('  radial counts here: ' + ', '.join(f'{e[i]:g}-{e[i + 1]:g}: {h[i]:,}' for i in range(len(h))))

    # 4. bound mask at this snapshot
    print('\n=== bound stars at this snapshot ===')
    pa = os.path.join(data_dir, 'particle-assignments')
    f_bound = os.path.join(pa, f'{tag}-bound-star-indices.npy')
    groups = None
    if os.path.exists(f_bound):
        groups = load_groups(f_bound)
        print(f'Alex\'s file {os.path.basename(f_bound)}: {len(groups):,} halos')
    else:
        try:
            say(f'no {os.path.basename(f_bound)}; reading the Rockstar catalogue to make it')
            hal = dipole.load_halo_catalog(args.sim, snap)
            hal = dipole.assign_particles_to_halo(sim=args.sim, snapshot=snap, part=part, halos=hal)
            groups = [np.asarray(x, np.int64) for x in hal['star.indices']]
        except Exception as ex:
            print(f'! no bound mask: {type(ex).__name__}: {ex}')
    keep_bound = np.ones(n_here, bool)
    if groups is not None:
        allidx = np.hstack([g for g in groups if len(g)])
        if allidx.max() >= n_here:
            sys.exit(f'bound file has index {allidx.max():,} >= {n_here:,} stars: not this snapshot')
        order = np.argsort([len(g) for g in groups])[::-1][:6]
        print('largest groups (halo index: N, distance of centre from host, median |x - centre|):')
        for i in order:
            x = xyz_all[groups[i]]
            cen = np.median(x, axis=0)
            spread = np.median(np.linalg.norm(x - cen, axis=1))
            d = np.linalg.norm(cen)
            flag = '  <- HOST? excluded' if d < 5 else ''
            print(f'  {i:7d}: {len(groups[i]):9,d}  {d:7.1f} kpc  {spread:6.2f} kpc{flag}')
            if d < 5:
                groups[i] = np.array([], np.int64)
        print('  (compact groups = star order of this file matches Alex\'s indexing)')
        bound_here = np.unique(np.hstack([g for g in groups if len(g)]))
        keep_bound[bound_here] = False
        print(f'bound stars: {len(bound_here):,}')
        print(f'Danny stars here that are bound: {np.isin(danny_here, bound_here).mean():.3f}')
        f600 = os.path.join(pa, f'{args.sim}-600-bound-star-indices.npy')
        if use_pointers and os.path.exists(f600):
            b600 = to_here(load_flat(f600), 'Alex bound at z = 0')
            print(f'  of these, bound here too: {np.isin(b600, bound_here).mean():.3f} '
                  '(a few tens of % or more expected; ~0 = pointers wrong)')

    nondh_here = None
    if args.nondh:
        print('\n=== Nondh (SAPFIRE) ===')
        nondh_here = to_here(sapfire_z0_indices(args.nondh, args.sim), 'nondh')
        print(f'  in Danny here: {np.isin(nondh_here, danny_here).mean():.3f}')

    # 5. save (same format as 02 + 05)
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
             redshift=part.snapshot['redshift'], snapshot=snap, source_file=args.file)
    say(f'saved {sel.sum():,} stars with r > {args.rmin_save} kpc to {out}')

    def keep_mask(inds):
        m = np.ones(n_here, bool)
        m[inds] = False
        return m[sel]

    masks = {'mask_danny': keep_mask(danny_here)}
    if gtr_here is not None:
        masks['mask_danny_gtr1e9'] = keep_mask(gtr_here)
    if nondh_here is not None:
        masks['mask_nondh'] = keep_mask(nondh_here)
    out2 = os.path.join(data_dir, f'{tag}-streammasks.npz')
    np.savez(out2, star_index=np.where(sel)[0], **masks)
    w = star['mass'][sel]
    for k, m in masks.items():
        print(f'  {k}: removes {(~m).sum():,} stars, {100 * w[~m].sum() / w.sum():.2f}% of the mass '
              f'beyond {args.rmin_save:g} kpc')
    print('saved', out2, '(True = keep)')
    print(f'\nnext: python scc/06_progressive_direct_m12i.py --sim {args.sim} --snap {snap}')


if __name__ == '__main__':
    main()
