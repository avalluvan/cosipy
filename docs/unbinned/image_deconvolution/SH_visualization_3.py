import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.special import sph_harm_y
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

import spherical
import quaternionic
from copy import deepcopy

c_kms    = 3e5       
N_SAMPLES = 100_000

THETA = None
PHI = None

def setup_grid(n_theta=120, n_phi=240):
    global THETA, PHI
    theta_1d = np.linspace(0, np.pi, n_theta)
    phi_1d = np.linspace(0, 2 * np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta_1d, phi_1d)
    print(f"Grid initialized: {n_theta}x{n_phi}")

# Initialize with defaults on import
setup_grid()

# ── Build surfaces ──
def spherical_to_cartesian(Y, THETA, PHI):
    return (Y * np.sin(THETA) * np.cos(PHI),
            Y * np.sin(THETA) * np.sin(PHI),
            Y * np.cos(THETA))

def get_rotation_matrix(theta_rot, phi_rot):
    # Target direction in Cartesian
    n = np.array([np.sin(theta_rot)*np.cos(phi_rot), np.sin(theta_rot)*np.sin(phi_rot), np.cos(theta_rot)])
    R, _ = Rotation.align_vectors([n], [[0, 0, 1]])     # Rotate z-axis to vec n
    return R

    # Equivalent code for axisymmetric terms using Euler rotations. However, it will break if non-axisymmetric terms are used. 
    # return Rotation.from_euler('yz', [theta_rot, phi_rot])

def rotate_sph_harm(l, m_source, R, THETA, PHI):
    """
    Rotate Y_l^{m_source} to a new axis defined by rotation matrix R,
    re-expressed on the original (THETA, PHI) grid via Wigner D-matrix.

    Returns
    -------
    Y_rot  : (THETA, PHI) ndarray  — rotated harmonic (real)
    X_rot, Y_rot_c, Z_rot          — rotated Cartesian surface
    """
    # ── Quaternion from rotation matrix ──
    x, y, z, w = R.as_quat()   # scipy: scalar-last → quaternionic: scalar-first
    quat = quaternionic.array([w, x, y, z])

    # ── Wigner D-matrix ──
    wig = spherical.Wigner(l)
    D   = wig.D(quat)

    # ── Rotated harmonic: Ỹ_l^0 = Σ_{m'} D^(l)*_{m',0} · Y_l^{m'} ──
    Y_rot = np.zeros_like(THETA, dtype=complex)
    for mp in range(-l, l+1):
        D_coeff = D[wig.Dindex(l, mp, m_source)].conj()
        Y_rot  += D_coeff * sph_harm_y(l, mp, THETA, PHI)
    Y_rot = Y_rot.real

    return Y_rot

# ── Mode Definition ──

def make_mode(l, m, theta_rot, phi_rot, amplitude=1.0):
    """
    Define a single spherical harmonic mode.
    
    Returns a dict describing the mode — not yet evaluated.
    """
    return dict(l=l, m=m, theta_rot=theta_rot, phi_rot=phi_rot, amplitude=amplitude)


def eval_mode(mode, THETA, PHI, sum_type):
    """
    Evaluate a single mode on the (THETA, PHI) grid.

    Parameters
    ----------
    sum_mode : 'signed'   — return raw signed values
               'unsigned' — mask negative values before returning
    """
    R = get_rotation_matrix(mode['theta_rot'], mode['phi_rot'])
    Y = mode['amplitude'] * rotate_sph_harm(mode['l'], mode['m'], R, THETA, PHI)

    if sum_type == 'unsigned':
        return np.where(Y > 0, Y, 0)   # mask per mode, before summing
    return Y


def sum_modes(modes, THETA, PHI, sum_type='signed'):
    """
    Sum a list of modes on the (THETA, PHI) grid.

    Parameters
    ----------
    modes : list of dicts from make_mode()
    sum_type: 'signed'   — sum raw values, raise error if result has negatives
              'unsigned' — mask each mode's negative lobes before summing
    """
    if sum_type not in ('signed', 'unsigned'):
        raise ValueError(f"Unknown mode '{sum_type}'. Choose 'signed' or 'unsigned'.")

    Y_sum = sum(eval_mode(m, THETA, PHI, sum_type=sum_type) for m in modes)

    if sum_type == 'signed' and np.any(Y_sum < 0):
        Y_00 = sph_harm_y(0, 0, 0.0, 0.0).real   # = 1/(2√π) ≈ 0.2821
        neg_frac = np.sum(Y_sum < 0) / Y_sum.size
        required_amplitude = -Y_sum.min() / Y_00
        raise ValueError(
            f"Y_sum has negative values at {neg_frac*100:.1f}% of (theta, phi) points. "
            f"Minimum value is {Y_sum.min():.4f}. "
            f"Increase the monopole (l=0, m=0) amplitude to at least {required_amplitude:.4f}."
        )

    return Y_sum

def get_doppler_inputs(density_modes, velocity_modes, THETA, PHI, sum_type='unsigned'):
    """
    Evaluate density and velocity fields from mode lists and compute
    sampling weights and line-of-sight velocity.

    Returns
    -------
    density : dict with keys 'Y', 'x', 'y', 'z'
    velocity : dict with keys 'Y', 'x', 'y', 'los'
    weights  : solid-angle weighted density, normalized
    """
    Y_den = sum_modes(density_modes, THETA, PHI, sum_type=sum_type)   # density field
    Y_vel = sum_modes(velocity_modes, THETA, PHI, sum_type=sum_type)  # velocity field

    rho_x, rho_y, rho_z = spherical_to_cartesian(Y_den, THETA, PHI)
    V_x,   V_y,   V_los = spherical_to_cartesian(Y_vel, THETA, PHI)   # Z_cart = line-of-sight

    weights  = Y_den * np.sin(THETA)                # solid-angle weighted density (full form would be sin(THETA) * dtheta * dphi but the latter are constants that will get normalized out)
    weights /= weights.sum()                        # normalize to probability

    density  = dict(Y=Y_den, x=rho_x, y=rho_y, z=rho_z)
    velocity = dict(Y=Y_vel, x=V_x,   y=V_y,   z=V_los)

    return density, velocity, weights

# ── Plotting ──

def plot_field(X, Y, Z, Y_sum, cmap='Blues', ex=0.5):
    fig, ax = plt.subplots(1, 1, figsize=(9, 9), subplot_kw={'projection': '3d'})

    sc = ax.scatter(X, Y, Z, s=1, c=Y_sum, cmap=cmap, norm=Normalize(vmin=0))
    ax.set_xlim(-ex, ex); ax.set_ylim(-ex, ex); ax.set_zlim(-ex, ex)

    plt.colorbar(sc, shrink=0.6)#, label=fr'Y$_{}$')
    return

def plot_two_fields(velocity, density, cmap='Blues', ex=0.5, show_morphology=False):
    Y_vel, V_x, V_y, V_los     = velocity['Y'], velocity['x'], velocity['y'], velocity['z']
    Y_den, rho_x, rho_y, rho_z = density['Y'], density['x'], density['y'], density['z']

    fig, axs = plt.subplots(1, 2, figsize=(18, 9), subplot_kw={'projection': '3d'})

    sc = axs[0].scatter(V_x, V_y, V_los, s=1, c=Y_vel, cmap=cmap, norm=Normalize(vmin=-0, vmax=Y_vel.max()))
    axs[0].set_xlim(-ex, ex); axs[0].set_ylim(-ex, ex); axs[0].set_zlim(-ex, ex)
    axs[0].set_title('Velocity')

    indices = ...#np.sort(np.random.choice(np.arange(len(V_x.ravel())), size=28800, replace=False))
    if show_morphology:
        sc = axs[1].scatter(V_x.ravel()[indices], V_y.ravel()[indices], V_los.ravel()[indices], s=1, c=Y_den.ravel()[indices], cmap=cmap, alpha=1, norm=Normalize(vmin=0, vmax=Y_den.max()))
        axs[1].set_title('Density (with Remnant Morphology)')       # Assumes homologous expansion of ejecta on prescribed velocity field
    else:
        X, Y, Z = spherical_to_cartesian(np.ones_like(THETA) / np.sqrt(4*np.pi), THETA, PHI)
        sc = axs[1].scatter(X.ravel()[indices], Y.ravel()[indices], Z.ravel()[indices], s=1, cmap=cmap,
                            c=Y_den.ravel()[indices], norm=Normalize(vmin=0, vmax=Y_den.max()))
        axs[1].set_title('Density (projected on Sphere)')

    axs[1].set_xlim(-ex, ex); axs[1].set_ylim(-ex, ex); axs[1].set_zlim(-ex, ex)

    plt.colorbar(sc, ax=axs, shrink=0.6)
    return

# ── Doppler Spectrum ──

def get_doppler_spectrum(V_los, Y_vel, weights, v_max=5000, sigma_kev=7.5/2.355, E_line_kev=1157.0, nbins=300, n_samples = N_SAMPLES):

    # ── Scale V_los to physical velocities ──
    # Normalize by the max of the velocity field, then scale to v_max
    V_los_physical = (V_los / Y_vel.max()) * v_max   # km/s

    # ── Convert keV smoothing to km/s ──
    # Doppler: ΔE/E = v/c  →  σ_v = (σ_E / E_0) * c                       # speed of light in km/s
    sigma_v  = (sigma_kev / E_line_kev) * c_kms  # km/s
    # print(f"Spectral smoothing: {sigma_kev} keV → {sigma_v:.1f} km/s")

    # ── Sample ──
    indices   = np.random.choice(V_los_physical.size, size=n_samples, p=weights.ravel())
    v_sampled = V_los_physical.ravel()[indices]

    # ── Histogram ──
    bins        = np.linspace(-12000, 12000, nbins)
    counts, edges = np.histogram(v_sampled, bins=bins)
    bin_centers   = 0.5 * (edges[:-1] + edges[1:])

    # ── Gaussian smoothing ──
    bin_width    = bin_centers[1] - bin_centers[0]   # km/s per bin
    sigma_bins   = sigma_v / bin_width               # convert km/s → bins
    counts_smooth = gaussian_filter1d(counts.astype(float), sigma=sigma_bins)

    return counts, edges, bin_centers, counts_smooth

def plot_doppler_spectrum(counts, edges, bin_centers, counts_smooth, v_max=5000, sigma_kev=7.5/2.355, E_line_kev=1157.0):
    # ── Plot ──
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.stairs(counts, edges, alpha=0.4, label='Raw')
    ax.plot(bin_centers, counts_smooth, label=f'Smoothed ({sigma_kev:.2f} keV)')
    ax.set_xlabel("Line-of-sight velocity (km/s)")
    ax.set_ylabel("Counts")
    ax.set_xlim(-12000, 12000)
    # ax2 = ax.twiny()
    # ax2.plot(E_line_kev + bin_centers / c_kms * E_line_kev, counts_smooth, label=f'Smoothed ({sigma_kev:.2f} keV)')
    ax.axvline(0, color='k', lw=1, ls='--')
    ax.legend()
    ax.set_title(f"Doppler Spectrum — 1157 keV line   v_max={v_max} km/s")
    plt.tight_layout()
    plt.show()

# ── Forward Model ──

# For each mode: [amplitude, theta_rot, phi_rot]
# l, m, sum_type are fixed model structure choices

def params_to_modes(params, l_den, l_vel):
    n_den = len(l_den)
    n_vel = len(l_vel)
    n_params_expected = 3 * (n_den + n_vel) + 1
    if len(params) != n_params_expected:
        raise ValueError(
            f"params length mismatch: got {len(params)}, "
            f"expected {n_params_expected} "
            f"(3×{n_den} density + 3×{n_vel} velocity + 1 v_max)."
        )

    density_modes, velocity_modes = [], []

    for i, l in enumerate(l_den):
        A, t, p = params[3*i : 3*i+3]
        density_modes.append(make_mode(l=l, m=0, theta_rot=t, phi_rot=p, amplitude=A))

    offset = 3 * n_den
    for j, l in enumerate(l_vel):
        A, t, p = params[offset + 3*j : offset + 3*j+3]
        velocity_modes.append(make_mode(l=l, m=0, theta_rot=t, phi_rot=p, amplitude=A))

    v_max = params[-1]
    return density_modes, velocity_modes, v_max

def forward_model(params, l_den, l_vel, sigma_kev=7.5/2.355, E_line_kev=1157.0, nbins=300):
    """Full forward pass: params → spectrum."""
    density_modes, velocity_modes, v_max = params_to_modes(params, l_den, l_vel)

    Y_den = sum_modes(density_modes, THETA, PHI, sum_type='unsigned')
    Y_vel = sum_modes(velocity_modes, THETA, PHI, sum_type='unsigned')

    _, _, V_los = spherical_to_cartesian(Y_vel, THETA, PHI)

    weights  = Y_den * np.sin(THETA)
    weights /= weights.sum()

    _, _, bin_centers, counts_smooth = get_doppler_spectrum(
        V_los, Y_vel, weights, v_max=v_max,
        sigma_kev=sigma_kev, E_line_kev=E_line_kev, nbins=nbins
    )
    
    return bin_centers, counts_smooth

def forward_model_pdf(params, l_den, l_vel, sigma_kev=7.5/2.355, E_line_kev=1157.0, nbins=300):
    """Full forward pass: params → spectrum."""
    density_modes, velocity_modes, v_max = params_to_modes(params, l_den, l_vel)

    Y_den = sum_modes(density_modes, THETA, PHI, sum_type='unsigned')
    Y_vel = sum_modes(velocity_modes, THETA, PHI, sum_type='unsigned')

    _, _, V_los = spherical_to_cartesian(Y_vel, THETA, PHI)

    weights  = Y_den * np.sin(THETA)
    weights /= weights.sum()

    _, _, bin_centers, counts_smooth = get_doppler_spectrum(
        V_los, Y_vel, weights, v_max=v_max,
        sigma_kev=sigma_kev, E_line_kev=E_line_kev, nbins=nbins
    )

    return interp1d(bin_centers, counts_smooth / counts_smooth.sum(), bounds_error=False, fill_value=1e-300)