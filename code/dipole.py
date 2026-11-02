# Adapted for BU SCC (/projectnb/fire-reflex, fire_env), Oct 2026.
# Changes from ahriley/dipoles (see SCC-CHANGES.md):
#  - paths: sim data under $FIRE_SIM_ROOT, outputs under $DIPOLE_DATA_DIR
#  - healpy, pandas, deepdish, astropy imported only inside the functions that
#    use them (fire_env lacks them; nothing in the basic fits needs them)
#  - current gizmo_analysis/halo_analysis API: snapshot kind 'index' (not
#    'snapshot'), assign_hosts_rotation=True, renamed assign_particle_indices kwargs
#  - np.bool -> bool
#  - NumPy 2 shims for np.cross (2-D) and np.linalg.eig (symmetric) used by utilities
import os
import numpy as np
from scipy.optimize import Bounds, minimize, basinhopping


# ---- NumPy >= 2 compatibility for Wetzel's utilities (fire_env has NumPy 2.5) ----
# 1. np.cross no longer accepts 2-component vectors; utilities.coordinate uses them
#    for v_phi (cylindrical) and for the disc-spin check in get_principal_axes.
#    The 2-D cross product is the scalar a_x b_y - a_y b_x, which is what old NumPy returned.
# 2. np.linalg.eig now returns complex arrays even for a real symmetric matrix, so the
#    inertia-tensor rotation comes out complex64 (ComplexWarning). For real symmetric
#    input use eigh, which is the correct solver there and returns real values;
#    utilities re-sorts the eigenvalues and fixes the axis signs itself afterwards.
# Both only change cases that NumPy 2 would otherwise reject or complexify.
_np_cross = np.cross
_np_eig = np.linalg.eig


def _cross_compat(a, b, *args, **kwargs):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape[-1] == 2 and b.shape[-1] == 2 and not args and not kwargs:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return _np_cross(a, b, *args, **kwargs)


def _eig_compat(a):
    arr = np.asarray(a)
    if (arr.ndim == 2 and np.isrealobj(arr)
            and np.allclose(arr, arr.T, rtol=1e-6, atol=0)):
        return np.linalg.eigh(arr)
    return _np_eig(a)


np.cross = _cross_compat
np.linalg.eig = _eig_compat
import utilities.coordinate as wcoord
import gizmo_analysis as gizmo
import halo_analysis as halo
from pygaia import proper_motion_uncertainty

SIM_ROOT = os.environ.get('FIRE_SIM_ROOT', '/projectnb/fire-reflex/latte')
DATA_DIR = os.environ.get('DIPOLE_DATA_DIR', '/projectnb/fire-reflex/rpei/data')

# tracer array is [relative distance err, vlos err, absolute magnitude]
# distances + vlos errors from Adrian. absmags are from:
# RGB: MIST isochrone, Fe/H = -1.5, age = 10 Gyr, logg ~= 1
# RRL: 2022MNRAS.513..788G (eqn 19, Fe/H = -1.5)
# del: test of this function, incredibly small errors
tracer_pars = {'rgb': [0.1, 5, -2.158428],
               'rrl': [0.05, 20, 0.555],
               'del': [0.0001, 0.0001, -10]
               }


def get_simulation_directory(sim):
    return os.path.join(SIM_ROOT, f'{sim}_res7100/')


def get_data_directory():
    return DATA_DIR


def get_rockstar_directory(sim):
    sim_dir = get_simulation_directory(sim)
    normal = os.path.exists(os.path.join(sim_dir, 'halo/rockstar_dm'))
    rockstar_dir = 'halo/rockstar_dm/' if normal else 'halo/rockstar_dm_new/'
    return rockstar_dir


def load_data(sim, snapshot, part_type=['star'], assign_pointers=False,
              sort_dark_by_id=False, snapshot_kind='index'):
    sim_dir = get_simulation_directory(sim)
    part = gizmo.io.Read.read_snapshots(part_type,
                                        snapshot_values=snapshot,
                                        snapshot_value_kind=snapshot_kind,
                                        simulation_directory=sim_dir,
                                        assign_pointers=assign_pointers,
                                        sort_dark_by_id=sort_dark_by_id,
                                        assign_hosts_rotation=True)
    halos = load_halo_catalog(sim, snapshot)
    return part, halos


def load_halo_catalog(sim, snapshot):
    sim_dir = get_simulation_directory(sim)
    rockstar_dir = get_rockstar_directory(sim)
    halos = halo.io.IO.read_catalogs('index', snapshot, sim_dir,
                                     rockstar_directory=rockstar_dir)
    return halos


def shell_radii(center, width):
    low = center - width/2
    high = center + width/2
    return low, high


def mask_distance_shell(distance, low, high):
    in_shell = np.logical_and(distance > low, distance < high)
    return in_shell


def mask_disk(distance):
    return distance > 5


def mask_halo_debris_at_z0(cat_index, part, hal, part_z0):
    pointers = part.Pointer.get_pointers(species_name_from='star',
                                         species_names_to='star', forward=True)
    indices_at_z = hal['star.indices'][cat_index]
    indices_at_z0 = pointers[indices_at_z]
    mask = np.ones_like(part_z0['star']['mass'], dtype=bool)
    mask[indices_at_z0] = 0
    return mask


def mask_nondh_streams_stars(stars, sim, mask_groups=None):
    if mask_groups is None:
        mask_groups = ['dwarf galaxy', 'coherent stream']
    sim_dir = get_simulation_directory(sim)
    streamfile = os.path.join(sim_dir, 'streams/all_groups.h5')
    import deepdish  # not in fire_env; may be replaceable by h5py
    groups = deepdish.io.load(streamfile)
    is_stream = np.isin(groups['classification'], mask_groups)
    stars_in_streams = np.unique(np.hstack(groups['st_indices'][is_stream]))
    mask = np.ones_like(stars['mass'], dtype=bool)
    mask[stars_in_streams] = 0
    return mask


def mask_danny_streams_stars(stars, sim):
    data_dir = get_data_directory()
    filename = f'mergers_{sim}_all.npy'
    filename = os.path.join(data_dir, 'danny-streams', filename)
    stars_in_streams = np.load(filename, allow_pickle=True)
    stars_in_streams = np.unique(np.hstack(stars_in_streams))
    mask = np.ones_like(stars['mass'], dtype=bool)
    mask[stars_in_streams] = 0
    return mask


def mask_m12i_messy_stream(stars):
    data_dir = get_data_directory()
    filename = 'messy_big_stream_m12i.txt'
    filename = os.path.join(data_dir, 'emily-streams', filename)
    stars_in_stream = np.loadtxt(filename, dtype=int)
    stars_in_stream = np.unique(np.hstack(stars_in_stream))
    mask = np.ones_like(stars['mass'], dtype=bool)
    mask[stars_in_stream] = 0
    return mask


def mask_bound_stars(stars, halos):
    not_host = np.where(np.sum(halos['host.distance']**2, axis=1) > 0)[0]
    stars_in_halos = np.unique(np.hstack(halos['star.indices'][not_host]))
    mask = np.ones_like(stars['mass'], dtype=bool)
    mask[stars_in_halos] = 0
    return mask


def assign_particles_to_halo(sim, snapshot, part_type='star',
                             part=None, halos=None, include_main=False,
                             data_dir=None):
    if data_dir is None:
        data_dir = get_data_directory()
        data_dir = os.path.join(data_dir, 'particle-assignments')
    filename = f'{sim}-{snapshot}-bound-{part_type}-indices.npy'
    filename = os.path.join(data_dir, filename)

    # if already computed, simply load
    if os.path.exists(filename):
        new_indices = np.load(filename, allow_pickle=True)
        halos[part_type+'.indices'] = new_indices
        return halos

    # set upper mass limit
    if include_main:
        mass_upper_lim = np.inf
    else:
        host_index = np.argmin(np.sum(halos['host.distance']**2, axis=1))
        mass_upper_lim = halos['mass'][host_index]

    # default kwargs
    kwargs = {'lowres_mass_frac_max': 1e-6,
              'mass_limits': [1e8, mass_upper_lim],
              'vel_circ_max_limits': [1, np.inf],
              'halo_radius_factor_max': 1,     # was halo_radius_frac_max
              'radius_max': 400,
              'velocity_factor_max': 2.0,      # was halo_velocity_frac_max
              'particle_number_fraction_converge': np.inf}

    particle_class = halo.io.ParticleClass()
    particle_class.assign_particle_indices(hal=halos, part=part,
                                           species=part_type, **kwargs)
    key = part_type + '.indices'
    new_indices = [np.array(idx_list).astype(int) for idx_list in halos[key]]
    new_indices = np.array(new_indices, dtype='object')
    halos[key] = new_indices
    os.makedirs(data_dir, exist_ok=True)
    np.save(filename, new_indices)
    return halos


def hpx_velocities(lon, lat, vels, nside):
    import healpy as hp
    import pandas as pd
    indices = hp.ang2pix(nside, lat, lon)
    npix = hp.nside2npix(nside)

    # use pandas groupby to compute mean in each healpix
    df = pd.DataFrame({'hpx': indices, 'v0': vels[:, 0], 'v1': vels[:, 1],
                       'v2': vels[:, 2]})
    means = df.groupby('hpx').mean()

    # create and fill in maps
    idx, counts = np.unique(indices, return_counts=True)
    hpx_maps = np.zeros((4, npix))
    hpx_maps[0, idx] = counts
    hpx_maps[1, means.index] = means['v0']
    hpx_maps[2, means.index] = means['v1']
    hpx_maps[3, means.index] = means['v2']
    hpx_maps = hp.ma(hpx_maps)

    # mask values
    mask = np.zeros_like(hpx_maps, dtype=bool)
    mask[hpx_maps == 0] = 1
    hpx_maps.mask = mask
    return hpx_maps


# largely recycled from Riley+2019
def lnlike(pars, xyz, v_sph, data_covs=None):
    v_sph_reflex = add_reflex_motion(pars[:3], v_sph, xyz)
    shifts = v_sph_reflex - pars[3:6]
    sigma = 10**pars[6:]
    if data_covs is None:
        # much simpler (and faster) when no observational errors
        lnlike = np.sum((shifts / sigma)**2)
        lnlike += np.log(np.prod(sigma**2)) * len(xyz)
    else:
        cov_theta = np.diag(sigma**2)
        covs = data_covs + cov_theta
        icovs = np.linalg.inv(covs)
        lnlike = np.sum([np.matmul(shift, np.matmul(icov, shift))
                        for shift, icov in zip(shifts, icovs)])
        lnlike += np.sum(np.log(np.linalg.det(covs)))

    # constants that were dropped in Riley+2019
    lnlike += 3*np.log(2*np.pi) * len(xyz)
    lnlike *= -0.5
    return lnlike


def lnprior(pars):
    # v_travel, theta, phi = pars[:3]
    reflex = pars[:3]
    mean = pars[3:6]
    # sigma = pars[6:]
    log_sigma = pars[6:]

    # in_prior = v_travel > 0
    # in_prior &= theta >= 0 and theta <= np.pi
    # in_prior &= phi >= 0 and phi < 2*np.pi
    in_prior = (reflex < 300).all() and (reflex > -300).all()
    in_prior &= (mean < 500).all() and (mean > -500).all()
    # in_prior &= (sigma < 300).all() and (sigma > 0).all()
    in_prior &= (log_sigma < 2.5).all() and (log_sigma > -2).all()

    return 0.0 if in_prior else -np.inf


def lnprob(pars, xyz, v_sph, data_covs=None):
    lp = lnprior(pars)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(pars, xyz, v_sph, data_covs)


def vel_xyz_to_sph_optimized(v_xyz, xyz):
    x, y, z = xyz.T
    vx, vy, vz = v_xyz.T
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    x2_y2 = x**2 + y**2

    v_sph = np.zeros_like(xyz)
    v_sph[:, 0] = (x*vx + y*vy + z*vz) / r
    v_sph[:, 1] = r*((z*(x*vx+y*vy)-vz*(x2_y2)) / (r**2*np.sqrt(x2_y2)))
    v_sph[:, 2] = r*np.sin(theta)*((x*vy - y*vx) / x2_y2)

    return v_sph


def add_reflex_motion(reflex_xyz, v_sph, xyz):
    # old way (parametrized reflex spherically instead of cartesian)
    # convert reflex to cartesian treating v_travel = reflex[0] as a distance
    # NOTE: this trick only works because reflex is defined *at the origin*
    # reflex_xyz = wcoord.get_positions_in_coordinate_system(reflex,
    #                                                        'spherical',
    #                                                        'cartesian')

    # old way (relied on wcoord + making lots of copies)
    # reflex_xyz = np.vstack([reflex_xyz] * len(xyz))
    # reflex_sph = wcoord.get_velocities_in_coordinate_system(reflex_xyz, xyz,
    #                                                         'cartesian',
    #                                                         'spherical')

    # Emily's hack to add reflex motion to spherical velocities
    reflex_sph = vel_xyz_to_sph_optimized(reflex_xyz, xyz)
    v_sph_reflex = v_sph + reflex_sph
    return v_sph_reflex


def add_uncertainties(xyz, v_sph):
    return xyz, v_sph


def generate_fake_data(pars, size, seed=None):
    # reflex_opposite = [-pars[0], pars[1], pars[2]]
    reflex_opposite = np.array([-pars[0], -pars[1], -pars[2]])
    mean = pars[3:6]
    sigma = 10**pars[6:]

    # xyz are a Gaussian ball centered on origin
    # _spherical_ vels are Gaussian ball w/ pars[3:]
    rng = np.random.default_rng(seed)
    xyz = rng.normal(loc=[0, 0, 0], scale=[50, 50, 50], size=size)
    v_sph = rng.normal(loc=mean, scale=sigma, size=size)

    # add reflex motion
    v_sph_reflex = add_reflex_motion(reflex_opposite, v_sph, xyz)

    return xyz, v_sph_reflex


def get_bounds():
    # lb = [0, 0, 0, -500, -500, -500, 0, 0, 0]
    # ub = [np.inf, np.pi, 2*np.pi, 500, 500, 500, 300, 300, 300]
    lb = [-300, -300, -300, -500, -500, -500, -2, -2, -2]
    ub = [300, 300, 300, 500, 500, 500, 2.5, 2.5, 2.5]
    bounds = Bounds(lb=lb, ub=ub)
    return bounds


def optimize_pars_local(initial, xyz, v_sph,
                        bounds=None, method='SLSQP', maxiter=100):
    bounds = get_bounds() if bounds is None else bounds
    soln = minimize(lambda *args: -lnlike(*args), initial,
                    args=(xyz, v_sph, None), bounds=bounds,
                    method=method, options={'maxiter': maxiter})
    return soln


def optimize_pars_global(initial, xyz, v_sph,
                         niter=100, bounds=None, minimizer_kwargs=None):
    bounds = get_bounds() if bounds is None else bounds
    if minimizer_kwargs is None:
        minimizer_kwargs = {'args': (xyz, v_sph, None),
                            'bounds': bounds,
                            'method': 'SLSQP',
                            'options': {'maxiter': 100}
                            }

    soln = basinhopping(lambda *args: -lnlike(*args), initial,
                        minimizer_kwargs=minimizer_kwargs, niter=niter)
    return soln


def transform_model_outputs(model):
    new = model.copy()

    # v_travel from cartesian to spherical
    new[:, :3] = wcoord.get_positions_in_coordinate_system(new[:, :3],
                                                           'cartesian',
                                                           'spherical')

    # log10(sigma) -> sigma
    new[:, 6:] = 10**new[:, 6:]
    return new


def track_halo_indices(tree, tree_index):
    # track backwards through time (main progenitor indices)
    prog_main_index = tree_index
    prog_main_indices = []
    while prog_main_index >= 0:
        prog_main_indices.append(prog_main_index)
        prog_main_index = tree['progenitor.main.index'][prog_main_index]

    # track forwards through time (descendants are unique)
    desc_index = tree_index
    desc_indices = []
    # TODO: there might be a better stopping criterion?
    while desc_index >= 0:
        desc_indices.append(desc_index)
        desc_index = tree['descendant.index'][desc_index]

    # rearrange so time progresses linearly (and no double-count tree_index)
    indices = prog_main_indices[::-1] + desc_indices[1:]
    return np.array(indices)


def get_z0_rotation_matrix(sim):
    sim_dir = get_simulation_directory(sim)
    # TODO: do I actually need to read particle data for this? seems overkill
    part = gizmo.io.Read.read_snapshots('star', 'redshift', 0, sim_dir,
                                        assign_hosts_rotation=True)
    rotation_matrix = part.host['rotation'][0]
    return rotation_matrix


def get_lmc_orbit(sim, tree_index, tree=None, rotation_matrix=None):
    sim_dir = get_simulation_directory(sim)

    if rotation_matrix is None:
        rotation_matrix = get_z0_rotation_matrix(sim)

    if tree is None:
        rockstar_dir = get_rockstar_directory(sim)
        tree = halo.io.IO.read_tree(simulation_directory=sim_dir,
                                    rockstar_directory=rockstar_dir)

    # get tree indices for main and LMC
    main_indices = track_halo_indices(tree, 0)
    indices = track_halo_indices(tree, tree_index)

    # identify snapshots where both exist
    snap = tree['snapshot'][indices]
    snap_main = tree['snapshot'][main_indices]
    sel, sel_main = np.intersect1d(snap, snap_main, return_indices=True)[1:]
    indices = indices[sel]
    main_indices = main_indices[sel_main]

    # determine when subhalo was within rvir
    rvir_host = tree['radius'][main_indices]
    dist = tree.prop('host.distance.total', indices)
    indices = indices[(dist > 0) & (dist < rvir_host)]

    # get positions wrt time (snapshots)
    xyz_boxcoords = tree.prop('host.distance', indices)
    xyz = wcoord.get_coordinates_rotated(xyz_boxcoords, rotation_matrix)
    snapshots = tree['snapshot'][indices]
    return snapshots, indices, xyz


def snapshot_to_time(sim, snapshots):
    sim_dir = get_simulation_directory(sim)
    data = np.loadtxt(sim_dir+'/snapshot_times.txt')
    time = data[snapshots, 3]
    return time


def sample_uncertainties(dist, tracer='rgb', release='dr3'):
    import astropy.units as u
    rng = np.random.default_rng()
    e_dist_rel, e_vlos_abs, abs_mag = tracer_pars[tracer]

    # distance errors (scaled)
    dist_scale_sampled = rng.normal(loc=1, scale=e_dist_rel, size=dist.shape)

    # vlos errors
    e_vlos_sampled = rng.normal(loc=0, scale=e_vlos_abs, size=dist.shape)

    dist_apy = dist * u.kpc

    # tangential errors from PyGaia (based on Gmag)
    gmag = abs_mag + 5 * np.log10(dist_apy / (10*u.pc))
    e_pmra, e_pmdec = proper_motion_uncertainty(gmag, release=release)
    e_pmra_sampled = rng.normal(loc=0, scale=e_pmra, size=dist.shape)
    e_pmdec_sampled = rng.normal(loc=0, scale=e_pmdec, size=dist.shape)
    e_vra_sampled = (e_pmra_sampled * u.microarcsecond / u.yr * dist_apy)
    e_vra_sampled = e_vra_sampled.to(u.km/u.s,
                                     equivalencies=u.dimensionless_angles())
    e_vdec_sampled = e_pmdec_sampled * u.microarcsecond / u.yr * dist_apy
    e_vdec_sampled = e_vdec_sampled.to(u.km/u.s,
                                       equivalencies=u.dimensionless_angles())

    result = (dist_scale_sampled, e_vlos_sampled, e_vra_sampled.value,
              e_vdec_sampled.value)
    return result


def mask_faint_tracers(dist, tracer='rgb', gmag_lim=20.7):
    import astropy.units as u
    abs_mag = tracer_pars[tracer][2]
    gmag = abs_mag + 5 * np.log10(dist*u.kpc / (10*u.pc))
    mask = gmag < gmag_lim
    return mask
