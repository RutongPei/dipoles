"""Step 4 (+ star counts for step 6): read m12i snapshot 600 on SCC, check the
host centring and principal-axis rotation, count star particles in Alex's
distance selections, and save the halo stars to a small .npz so the dipole
fits never have to re-read the 7.7 GB snapshot.

Run on a compute node (needs ~20-30 GB RAM), from the cloned repo's top level:
    cd /projectnb/fire-reflex/rpei/dipoles
    python scc/02_read_m12i_600.py                 # Rockstar bound-star mask
    python scc/02_read_m12i_600.py --alex-bound    # + Alex's re-assignment (slower)

Output: $DIPOLE_DATA_DIR/m12i-600-halostars.npz  (default /projectnb/fire-reflex/rpei/data)
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
import dipole  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--sim', default='m12i')
ap.add_argument('--snap', type=int, default=600)
ap.add_argument('--rmin-save', type=float, default=20., help='save stars beyond this r [kpc]')
ap.add_argument('--alex-bound', action='store_true',
                help="also run Alex's assign_particles_to_halo (his poster cuts)")
args = ap.parse_args()
t0 = time.time()


def say(*a):
    print(f'[{time.time() - t0:7.1f} s]', *a, flush=True)


# ---------------------------------------------------------------- read -----
say(f'reading {args.sim} snapshot {args.snap} stars from', dipole.get_simulation_directory(args.sim))
part, hal = dipole.load_data(args.sim, args.snap)   # assign_hosts_rotation=True inside
star = part['star']
say(f'{len(star["mass"]):,} star particles read')

# ---------------------------------------------------------- host checks ----
print('\n=== host ===')
print('host position [kpc comoving box]:', np.round(part.host['position'][0], 2))
print('host velocity [km/s]            :', np.round(part.host['velocity'][0], 2))
print('principal axis ratios (min/maj, min/int, int/maj):',
      np.round(part.host['axis.ratios'][0], 3))
R = part.host['rotation'][0]
print('rotation matrix (rows = major, intermediate, minor axis):\n', np.round(R, 4))

xyz_all = star.prop('host.distance.principal').astype(np.float64)
vxyz_all = star.prop('host.velocity.principal').astype(np.float64)
r_all = np.sqrt(np.sum(xyz_all**2, axis=1))

# disc check: stellar angular momentum within 15 kpc should point along +/- z (minor axis)
inner = r_all < 15
L = np.sum(star['mass'][inner, None] * np.cross(xyz_all[inner], vxyz_all[inner]), axis=0)
cos_Lz = L[2] / np.linalg.norm(L)
Rcyl = np.hypot(xyz_all[:, 0], xyz_all[:, 1])
disc = (Rcyl > 4) & (Rcyl < 12) & (np.abs(xyz_all[:, 2]) < 2)
vphi_cyl = (xyz_all[disc, 0] * vxyz_all[disc, 1] - xyz_all[disc, 1] * vxyz_all[disc, 0]) / Rcyl[disc]
print(f'\ncos(angle between stellar L(<15 kpc) and principal z) = {cos_Lz:+.4f}  '
      '(want |value| close to 1)')
print(f'disc stars (4<R<12, |z|<2 kpc): N = {disc.sum():,}, median v_phi = {np.median(vphi_cyl):+.1f} km/s, '
      f'median |v_z| = {np.median(np.abs(vxyz_all[disc, 2])):.1f} km/s')
print('  (sign of v_phi only says which way the disc spins in these axes; |v_phi| ~ 200+ is the check)')

# spherical velocities from Alex's converter in float64 (same convention as Wetzel's
# '.spherical'; that one only runs thanks to the np.cross shim in dipole.py) -- cross-check here
v_sph_all = dipole.vel_xyz_to_sph_optimized(vxyz_all, xyz_all)
try:
    test = star.prop('host.velocity.principal.spherical', np.arange(100000))
    d = np.abs(test - v_sph_all[:100000])
    print(f"\nprop('host.velocity.principal.spherical') vs Alex's converter (first 1e5 stars): "
          f"median |diff| {np.median(d, axis=0)}, max {d.max(axis=0)} km/s "
          "(max can be large only for stars on the z-axis, where v_phi is ill-defined)")
except Exception as e:
    print(f"\nprop('host.velocity.principal.spherical') FAILS ({type(e).__name__}: {e});"
          " using dipole.vel_xyz_to_sph_optimized instead (identical convention)")

# ------------------------------------------------------- bound masks -------
print('\n=== bound-star masks ===')
masks = {'none': np.ones(len(r_all), bool)}
if 'star.indices' in hal:
    masks['bound_rockstar'] = dipole.mask_bound_stars(star, hal)
    print(f"Rockstar catalogue (star_{args.snap}.hdf5): "
          f"{(~masks['bound_rockstar']).sum():,} stars in non-host halos")
else:
    print('WARNING: halo catalogue has no star.indices (star_*.hdf5 not read)')
if args.alex_bound:
    try:
        say("running Alex's assign_particles_to_halo ...")
        hal_alex = dipole.assign_particles_to_halo(sim=args.sim, snapshot=args.snap,
                                                   part=part, halos=hal)
        masks['bound_alex'] = dipole.mask_bound_stars(star, hal_alex)
        print(f"Alex's re-assignment: {(~masks['bound_alex']).sum():,} stars in non-host halos")
    except Exception as e:
        print(f"Alex's re-assignment FAILED: {type(e).__name__}: {e}")

# --------------------------------------------------------- counts ----------
# Poisson-only floor from fake_data_experiments.py (sigma = 100, 80, 80 km/s, 50 realisations):
# median fitted |v_travel| when the true dipole is zero
N_TAB = np.array([100, 300, 1000, 3000, 20000])
FLOOR_TAB = np.array([16., 12., 5., 3., 1.])


def floor(n):
    if n < N_TAB[0]:
        return np.nan
    if n > N_TAB[-1]:                      # extrapolate as N^-1/2
        return FLOOR_TAB[-1] * np.sqrt(N_TAB[-1] / n)
    return 10**np.interp(np.log10(n), np.log10(N_TAB), np.log10(FLOOR_TAB))


centers = np.arange(30, 301, 10)
width, high = 5, 305
print('\n=== star-particle counts in the fit selections (Alex: centres 30-300, 5 kpc shells; '
      f'complementary = c < r < {high}) ===')
print('floor = median |v_travel| from pure Poisson noise at that N (fake-data table, km/s)\n')
hdr = f"{'centre':>6} | " + ' | '.join(f'{k + " binned":>22} {k + " compl.":>22}' for k in masks)
print(hdr)
counts = {}
for k, m in masks.items():
    counts[k] = {'binned': [], 'complementary': []}
    for c in centers:
        counts[k]['binned'].append(np.sum(m & (r_all > c - width / 2) & (r_all < c + width / 2)))
        counts[k]['complementary'].append(np.sum(m & (r_all > c) & (r_all < high)))
for i, c in enumerate(centers):
    row = f'{c:6d} | '
    row += ' | '.join(f"{counts[k]['binned'][i]:>12,} ({floor(counts[k]['binned'][i]):4.1f})  "
                      f"{counts[k]['complementary'][i]:>12,} ({floor(counts[k]['complementary'][i]):4.1f})"
                      for k in masks)
    print(row)

# ------------------------------------------------------------ save ---------
out_dir = dipole.get_data_directory()
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, f'{args.sim}-{args.snap}-halostars.npz')
sel = r_all > args.rmin_save
np.savez(out,
         xyz=xyz_all[sel].astype(np.float32),
         v_xyz=vxyz_all[sel].astype(np.float32),
         v_sph=v_sph_all[sel].astype(np.float32),
         dist=r_all[sel].astype(np.float32),
         mass=star['mass'][sel].astype(np.float32),
         feh=star.prop('metallicity.iron')[sel].astype(np.float32),
         star_index=np.where(sel)[0],          # index into the full snapshot star array
         **{f'mask_{k}': m[sel] for k, m in masks.items() if k != 'none'},
         host_position=part.host['position'][0], host_velocity=part.host['velocity'][0],
         host_rotation=R, host_axis_ratios=part.host['axis.ratios'][0])
say(f'saved {sel.sum():,} stars with r > {args.rmin_save} kpc to {out} '
    f'({os.path.getsize(out) / 1e6:.0f} MB)')
