"""Build the stream masks for m12i snapshot 600 on the saved halo stars,
and check that each list uses the same star indexing as m12i-600-halostars.npz.

Sources (all z = 0 star array indices, read from $DIPOLE_DATA_DIR):
    messy  : emily-streams/messy_big_stream_m12i.txt        (Emily's messy big stream)
    danny  : danny-streams/mergers-m12i-all.npy               (Danny streams; selection unconfirmed)
    nondh  : --nondh FILE (optional) from the SAPFIRE catalogue (Panithanpaisal+21)
             .npy / .txt = plain list of star indices;
             .csv = needs an 'index' column; rows are kept only where simname contains the sim name
             (if a 'simname' column exists). Add --nondh-xyz-check if the csv also has x, y, z.

Run from the repo top level (login node is fine; needs ~2 GB RAM, no snapshot read):
    python scc/05_stream_masks_m12i.py
    python scc/05_stream_masks_m12i.py --nondh /projectnb/fire-reflex/rpei/data/sapfire/m12i.csv

Output: $DIPOLE_DATA_DIR/m12i-600-streammasks.npz
    star_index (same order as the halostars npz), mask_messy, mask_danny[, mask_nondh]
    Masks follow Alex's convention: True = keep, False = star is in that structure.

Also, if particle-assignments/m12i-600-bound-star-indices-alex2021.npy exists, compares it with
the bound-star file 02_read_m12i_600.py --alex-bound made (same name without -alex2021).
"""
import argparse
import csv
import os
import sys

import numpy as np

DATA_DIR = os.environ.get('DIPOLE_DATA_DIR', '/projectnb/fire-reflex/rpei/data')


def danny_indices(data_dir, sim):
    a = np.load(os.path.join(data_dir, 'danny-streams', f'mergers-{sim}-all.npy'), allow_pickle=True)
    return np.unique(np.hstack(a)).astype(np.int64), len(np.hstack(a))


def messy_indices(data_dir):
    a = np.loadtxt(os.path.join(data_dir, 'emily-streams', 'messy_big_stream_m12i.txt'), dtype=np.int64)
    return np.unique(a), len(a)


def nondh_indices(path, sim):
    """Returns (unique indices, raw count, xyz or None)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        a = np.load(path, allow_pickle=True)
        a = np.hstack(a).astype(np.int64)
        return np.unique(a), len(a), None
    if ext == '.txt':
        a = np.loadtxt(path, dtype=np.int64)
        return np.unique(a), len(a), None
    if ext == '.csv':
        idx, xyz = [], []
        with open(path) as f:
            rd = csv.DictReader(f)
            cols = {c.lower(): c for c in rd.fieldnames}
            if 'index' not in cols:
                sys.exit(f'{path}: no "index" column (columns: {rd.fieldnames})')
            has_xyz = all(k in cols for k in ('x', 'y', 'z'))
            for r in rd:
                if 'simname' in cols and sim not in r[cols['simname']]:
                    continue
                idx.append(int(float(r[cols['index']])))
                if has_xyz:
                    xyz.append([float(r[cols[k]]) for k in ('x', 'y', 'z')])
        idx = np.array(idx, dtype=np.int64)
        u, first = np.unique(idx, return_index=True)
        return u, len(idx), (np.array(xyz)[first] if xyz else None)
    sys.exit(f'{path}: unsupported extension {ext} (use .npy, .txt or .csv)')


def compare_bound_files(data_dir, sim, snap):
    d = os.path.join(data_dir, 'particle-assignments')
    f_mine = os.path.join(d, f'{sim}-{snap}-bound-star-indices.npy')
    f_alex = os.path.join(d, f'{sim}-{snap}-bound-star-indices-alex2021.npy')
    if not (os.path.exists(f_mine) and os.path.exists(f_alex)):
        print(f'\n(no bound-file comparison: need both {os.path.basename(f_mine)} and '
              f'{os.path.basename(f_alex)})')
        return
    mine = np.load(f_mine, allow_pickle=True)
    alex = np.load(f_alex, allow_pickle=True)
    print('\n=== bound-star assignment: mine (Oct 2026) vs Alex (2021) ===')
    print(f'halos in catalogue: mine {len(mine):,}, Alex {len(alex):,}')
    nonempty = lambda arr: sum(1 for x in arr if len(x) > 0)
    print(f'halos with >= 1 star: mine {nonempty(mine):,}, Alex {nonempty(alex):,}')
    um = np.unique(np.hstack([x for x in mine if len(x)]).astype(np.int64))
    ua = np.unique(np.hstack([x for x in alex if len(x)]).astype(np.int64))
    both = np.intersect1d(um, ua)
    print(f'bound stars: mine {len(um):,}, Alex {len(ua):,}, both {len(both):,}, '
          f'only mine {len(um) - len(both):,}, only Alex {len(ua) - len(both):,}')
    if len(mine) == len(alex):
        diff = np.array([len(np.setxor1d(np.asarray(a, np.int64), np.asarray(b, np.int64)))
                         for a, b in zip(mine, alex)])
        order = np.argsort(diff)[::-1][:10]
        if diff[order[0]] == 0:
            print('per-halo star lists identical.')
        else:
            print('halos with the largest differences (halo index: N mine, N Alex, N differing):')
            for i in order:
                if diff[i] == 0:
                    break
                print(f'  {i:7d}: {len(mine[i]):7,d} {len(alex[i]):7,d} {diff[i]:7,d}')
    else:
        print('different halo counts: catalogues differ, so per-halo comparison is not meaningful.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', default='m12i')
    ap.add_argument('--snap', type=int, default=600)
    ap.add_argument('--nondh', default=None, help='SAPFIRE star list for this host (.npy/.txt/.csv)')
    ap.add_argument('--data-dir', default=DATA_DIR)
    args = ap.parse_args()
    tag = f'{args.sim}-{args.snap}'

    d = np.load(os.path.join(args.data_dir, f'{tag}-halostars.npz'))
    sidx = d['star_index'].astype(np.int64)
    dist = d['dist'].astype(np.float64)
    xyz = d['xyz'].astype(np.float64)
    mass = d['mass'].astype(np.float64)
    keep_bound = d['mask_bound_alex'] if 'mask_bound_alex' in d.files else None
    print(f'{tag}-halostars.npz: {len(sidx):,} stars with r > {dist.min():.1f} kpc, '
          f'star_index {sidx.min():,}..{sidx.max():,}')

    sets = {}
    u, n_raw = messy_indices(args.data_dir)
    sets['messy'] = (u, n_raw, None)
    u, n_raw = danny_indices(args.data_dir, args.sim)
    sets['danny'] = (u, n_raw, None)
    if args.nondh:
        sets['nondh'] = nondh_indices(args.nondh, args.sim)

    # ---- per-source checks ----
    print('\n=== sources ===')
    print('in_npz = fraction of the listed stars found among the saved r > 20 kpc stars.\n'
          'Stars not found are either inside 20 kpc or not valid indices; '
          'a value near 0 means the indexing does not match.')
    masks = {}
    M_all = mass.sum()
    for k, (u, n_raw, cat_xyz) in sets.items():
        inn = np.isin(sidx, u)
        masks[k] = ~inn
        frac = inn.sum() / len(u)
        print(f'\n[{k}] {n_raw:,} entries, {len(u):,} unique, index range {u.min():,}..{u.max():,}')
        print(f'  in_npz {inn.sum():,} ({100 * frac:.1f}%), mass fraction of r > 20 kpc stars '
              f'{100 * mass[inn].sum() / M_all:.2f}%')
        if inn.sum():
            h, e = np.histogram(dist[inn], bins=[20, 50, 100, 150, 200, 250, 300, 400, np.inf])
            print('  radial counts: ' + ', '.join(f'{int(e[i])}-{e[i + 1]:g}: {h[i]:,}'
                                                for i in range(len(h))).replace('-inf', '+'))
        if cat_xyz is not None:
            # catalogue frame may differ from ours: compare distances from the host only
            pos = {s: i for i, s in enumerate(sidx)}
            j = np.array([pos.get(s, -1) for s in u])
            ok = j >= 0
            if ok.sum():
                dr = np.linalg.norm(cat_xyz[ok], axis=1) - dist[j[ok]]
                print(f'  catalogue vs our distance (matched by index, N = {ok.sum():,}): '
                      f'median |dr| {np.median(np.abs(dr)):.3f} kpc, 90th pct '
                      f'{np.percentile(np.abs(dr), 90):.3f} kpc  (should be << 1 kpc if the index matches)')

    # ---- overlaps (within the saved stars) ----
    groups = {}
    if keep_bound is not None:
        groups['bound_alex'] = ~keep_bound
    groups.update({k: ~m for k, m in masks.items()})
    names = list(groups)
    print('\n=== overlaps among saved stars (row: N in row structure that are also in column) ===')
    print(' ' * 12 + ''.join(f'{n:>12s}' for n in names))
    for a in names:
        print(f'{a:>12s}' + ''.join(f'{(groups[a] & groups[b]).sum():12,d}' for b in names))
    if keep_bound is not None:
        for k in masks:
            n_in = (~masks[k]).sum()
            if n_in:
                print(f'{k}: {100 * (~masks[k] & ~keep_bound).sum() / n_in:.1f}% already removed by the bound mask')

    out = os.path.join(args.data_dir, f'{tag}-streammasks.npz')
    np.savez(out, star_index=sidx, **{f'mask_{k}': m for k, m in masks.items()})
    print('\nsaved', out, '(True = keep)')

    compare_bound_files(args.data_dir, args.sim, args.snap)


if __name__ == '__main__':
    main()
