# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cycler import cycler

import astropy.units as u
from astropy.units import Quantity
import astropy.constants as const
from astropy.coordinates import SkyCoord
from scipy import integrate
from scipy.special import erfc, erf
from scipy.stats import gaussian_kde

import os
import subprocess
from tqdm import tqdm
import multiprocess as mp
from pathlib import Path
import itertools

from histpy import Histogram, Axes, Axis, HealpixAxis
import healpy as hp
from mhealpy import HealpixMap, HealpixBase
from scoords import Attitude, SpacecraftFrame

from cosipy import UnBinnedData, BinnedData, test_data
from cosipy.response import FullDetectorResponse, DetectorResponse, PointSourceResponse
from cosipy.spacecraftfile import SpacecraftFile
from FullDetectorResponseNew import FullDetectorResponse
from DetectorResponseNew import DetectorResponse

from astromodels import PointSource, Parameter
from threeML import Model, Powerlaw, Gaussian, Constant, Band
from threeML import PluginPrototype, Model, JointLikelihood, DataList
from threeML.utils.statistics.likelihood_functions import poisson_log_likelihood_ideal_bkg

DATA_DIR = Path('/Users/penguin/Documents/Research/COSI/COSIpy/docs/DC3/') 
response_path = DATA_DIR / 'data/responses/Response44Ti.o4.e1154_1160.s9607532021290.m1215.filtered.nonsparse.binnedimaging.imagingresponse_nside16.area.h5' 

# %% 
# --- Configure Plot Style ---
def configure_plot_style():
    defaultcolor = '#002060'
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        'text.color': defaultcolor, 'axes.labelcolor': defaultcolor,
        'xtick.color': defaultcolor, 'ytick.color': defaultcolor,
        'axes.prop_cycle': cycler(color=['orange', 'limegreen', 'b', 'r']),
        'font.family': 'serif', 'font.serif': 'Times New Roman',
        'font.size': 22, 'lines.linewidth': 3,
        'figure.figsize': (9.6, 5.4), 'figure.dpi': 100
    })

# %%
def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

# %% 
# --- Load Detector Response ---
def load_detector_response(response_path, pix=0):
    with FullDetectorResponse.open(response_path) as f:
        # print(f.__dict__)
        dr = f[pix]
    return dr

# %% 
# --- Load Source Data ---
def load_data(DATA_DIR, name='CasAfullyresolved'):
    point_source = UnBinnedData(DATA_DIR / '44Ti/inputs.yaml')
    data = point_source.get_dict_from_fits(DATA_DIR / f'44Ti/sources/{name}_3months_unbinned_data_filtered_with_SAAcut.fits')
    target_coord = SkyCoord.from_name('Cas A').galactic
    return data, target_coord.l.value, target_coord.b.value

# %% 
# --- Load Background Data ---
def load_background(DATA_DIR):
    bg = UnBinnedData(DATA_DIR / '44Ti/inputs.yaml')
    bg_filepath = DATA_DIR / 'data/backgrounds/unbinned/AlbedoPhotons_44Ti.fits'
    bg_dict = bg.get_dict_from_fits(bg_filepath)
    return bg_dict

# %% 
# --- Estimate KDE from Background Energies ---
def estimate_kde(bg_dict):
    return gaussian_kde(bg_dict['Energies'])

# %% 
# --- Compute Rotated Background KDE ---
def compute_background_kde(bg_dict, rot_custom):
    bg_ang_rot = rot_custom(bg_dict['Chi galactic'], bg_dict['Psi galactic'], lonlat=True)
    background_data = np.array([bg_dict['Energies'], bg_dict['Phi'], bg_ang_rot[1], bg_ang_rot[0]])
    background_kde = gaussian_kde(background_data[[0,1,3],:])
    return background_kde, background_data

# %% 
# --- Sample Events from Data and Background ---
def sample_events(data, bg_dict, num_samples=1000, bgcounts=0):
    indices = np.random.choice(np.arange(len(bg_dict['Energies'])), bgcounts, replace=False)
    bg_energy_samples = bg_dict['Energies'][indices] * u.keV
    bg_phi_samples = bg_dict['Phi'][indices] * u.rad
    bg_psi_gal_samples = bg_dict['Psi galactic'][indices] * u.deg
    bg_chi_gal_samples = bg_dict['Chi galactic'][indices] * u.deg

    indices = np.random.choice(np.arange(len(data['Energies'])), num_samples, replace=False)
    energy_samples = np.concatenate([data['Energies'][indices] * u.keV, bg_energy_samples])
    phi_samples = np.concatenate([data['Phi'][indices] * u.rad, bg_phi_samples])
    psi_gal_samples = np.concatenate([data['Psi galactic'][indices] * u.deg, bg_psi_gal_samples])
    chi_gal_samples = np.concatenate([data['Chi galactic'][indices] * u.deg, bg_chi_gal_samples])

    return energy_samples, phi_samples, psi_gal_samples, chi_gal_samples

# %% 
# --- Plot Histogram of Sampled Energies ---
def plot_energy_histogram(energy_samples, data):
    bins = np.logspace(2, 3.3, 100)*u.keV
    plt.hist(energy_samples, bins=bins, density=True, label=f'{len(energy_samples):.1e} sample')
    plt.hist(data['Energies'], bins=bins.value, density=True, lw=2, histtype='step', label='Data')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Count Density')
    plt.xscale('log')
    plt.legend()
    plt.show()

# %% 
# --- Analyze Healpix Coordinates from Sampled Events ---
def analyze_healpix_coordinates(chi_gal_samples, psi_gal_samples, l, b, nside=32):
    rot_custom = hp.Rotator(rot=[l, b - 90], inv=False)
    ang_rot = rot_custom(chi_gal_samples.value, psi_gal_samples.value, lonlat=True)

    Psi_sc_onaxis = ang_rot[1] * u.deg  # b
    Chi_sc_onaxis = ang_rot[0] * u.deg  # l

    pixels_rot = hp.ang2pix(nside=nside, theta=ang_rot[0], phi=ang_rot[1], nest=False, lonlat=True)
    return Psi_sc_onaxis, Chi_sc_onaxis, pixels_rot, nside

# %% 
# --- Plot Healpix Map from Pixel Data ---
def plot_healpix_map(pixels_rot, nside=32):
    m_rot = HealpixMap(nside=nside, scheme='RING', dtype=int)
    for p in pixels_rot:
        m_rot[p] += 1
    m_rot.plot()
    plt.show()


# %% 
# --- Interpolation Weight Calculation ---
def get_all_interp_weights(dr, target: dict):
    indices, weights = [], []
    for label in dr.axes.labels:
        if label not in ['Phi', 'PsiChi']:
            raise ValueError(f'Label: {label} is not supported')
        axis = dr.axes[label]
        if isinstance(axis, (Axis, HealpixAxis)):
            if label == 'Phi':
                idx, w = axis.interp_weights(target[label])
            elif label == 'PsiChi':
                coord = SkyCoord(target['Chi_sc'], target['Psi_sc'], frame=SpacecraftFrame())
                idx, w = axis.interp_weights(coord)
        else:
            raise ValueError(f'Unsupported axis type: {type(axis)}')
        indices.append(idx)
        weights.append(w)
    return indices, weights

# %% 
# --- Interpolated Response Calculation ---
def get_interp_response(dr, target: dict):
    indices, weights = get_all_interp_weights(dr, target)
    perm_indices = list(itertools.product(*indices))
    perm_weights = list(itertools.product(*weights))
    interpolated_response_value = sum(np.prod(w) * dr.contents[idx] for idx, w in zip(perm_indices, perm_weights))
    return interpolated_response_value

# %% 
# --- Project Detector Response onto Phi and PsiChi Axes ---
def project_detector_response(dr):
    hist = dr.project('Phi', 'PsiChi')
    dr2 = DetectorResponse(coord=dr.coord, edges=hist.axes, contents=hist.contents, unit=dr.unit)
    return dr2

# %%
class COSILikeNew(PluginPrototype):

    def __init__(self, name, dr, exposure_time, energy_samples, 
                 phi_samples, psi_samples, chi_samples, 
                 spectrum_unit, nuisance_param, 
                 Eiedges, 
                 sigma_e, 
                 tau_e, 
                 energy_redistribution_kernel,
                 background_kde,
                 bgcounts):
        
        super().__init__(name, nuisance_param)  # Initialize plugin base class with name and nuisance parameters

        # Define constants
        self._s2 = np.sqrt(2)
        self._s2pi = np.sqrt(2 * np.pi)
        self._tiny = 1e-12

        # Raw detector response and observation metadata
        self._dr = dr
        self._exposure_time = exposure_time.to('s')

        # Observed event-level data
        self._energy_samples = energy_samples              # Measured energies
        self._phi_samples = phi_samples                    # Scattering angles
        self._psi_samples = psi_samples                    
        self._chi_samples = chi_samples                    

        # Spectral integration boundaries and resolution
        self._Eiedges = Eiedges                            # True energy range to integrate over. For time being, assumed to be the measured energy range too.
        self._delta_Ei = Eiedges[-1] - Eiedges[0]          # Width of true energy integration window
        self._delta_Em = Eiedges[-1] - Eiedges[0]          # Width of measured energy range (used for convolution)

        self._spectrum_unit = spectrum_unit                # Unit of spectral flux (e.g., ph/cm²/s/keV)
        self._n_events = len(energy_samples)               # Total number of observed events
        self._sigma_e = sigma_e                            # Gaussian energy resolution
        self._tau_e = tau_e                             # Expgaussian tailing
        self._tau_1 = 0.636 * u.keV                     # Double expgaussian tailing. Hardcoded based on empirical modeling of 1157 keV line. May change for other line energies.
        self._tau_2 = 12.5 * u.keV 
        self._w1 = 0.7626                               # Relative weights of double expgaussian. May change for other line energies. 

        self._effective_area = np.sum(self._dr.contents)           # Effective area of the detector hardcoded for line fitting
        self._energy_redistribution_kernel = energy_redistribution_kernel                # If None, then we are effectively modeling a detector with infinite energy resolution
        self._background_kde = background_kde
        self._bgcounts = bgcounts

    def gaus_kernel(self, Ei, Em, sigma_e):
        # Gaussian energy dispersion kernel
        # Probability of detecting Em given true energy Ei
        return (1 / (self._s2pi * sigma_e)) * np.exp(-((Em - Ei)**2) / (2 * sigma_e**2))

    def integrated_gaus_kernel(self, Ei, Eiedges, sigma_e):
        # Analytical integration of the gaussian kernel from E_min to E_max
        # Probability that an event with true energy Ei lands within measured range [E_min, E_max]
        return 0.5 * (erfc((Eiedges[0] - Ei) / (self._s2 * sigma_e)) -
                erfc((Eiedges[-1] - Ei) / (self._s2 * sigma_e)))
    
    def expgaus_kernel(self, Ei, Em, sigma_e, tau_e):    
        # Exponentially modified gaussian energy dispersion kernel
        gss = -sigma_e * sigma_e / tau_e
        arg1 = -(Ei + gss/2.0 - Em) / tau_e
        arg2 = (Ei + gss - Em)/max((self._tiny * sigma_e.unit), (self._s2 * sigma_e))
        return 1 / (2 * tau_e) * np.exp(arg1) * (erf(arg2) + 1)
    
    def integrated_expgaus_kernel(self, Ei, Eiedges, sigma_e, tau_e):
        # Numerically integrate the expgaussian kernel from E_min to E_max
        redistribution_kernel = lambda Em: self.expgaus_kernel(Ei, Em * u.keV, sigma_e, tau_e) * u.keV      # This is here instead of the next line as the integral will fail otherwise

        result = integrate.quad(redistribution_kernel, Eiedges[0].value, Eiedges[-1].value)[0]
        return result
    
    def double_expgaus_kernel(self, Ei, Em, sigma_e, w1, tau_1, tau_2):
        return w1 * self.expgaus_kernel(Ei, Em, sigma_e, tau_1) + \
               (1 - w1) * self.expgaus_kernel(Ei, Em, sigma_e, tau_2)
    
    def integrated_double_expgaus_kernel(self, Ei, Eiedges, sigma_e, w1, tau_1, tau_2):
        # Numerically integrate the double expgaussian kernel from E_min to E_max
        redistribution_kernel = lambda Em: self.double_expgaus_kernel(Ei, Em * u.keV, sigma_e, 
                                                                      w1, tau_1, tau_2) * u.keV

        result = integrate.quad(redistribution_kernel, Eiedges[0].value, Eiedges[-1].value, 
                                epsabs=1e-6, epsrel=1e-6)[0]
        return result

    def set_model(self, likelihood_model):
        # Set the spectral model (typically ThreeML source model)
        self._likelihood_model = likelihood_model

    def _compute_signal_density(self, Em, Phi, Psi, Chi, Ei_min, Ei_max, redistribution_kernel, spectrum):
        
        if redistribution_kernel is None:
            dispersed_spectrum = spectrum
        else:
            dispersed_spectrum = lambda Ei: redistribution_kernel(Ei, Em)

        integrated_flux = integrate.quad(dispersed_spectrum, Ei_min, Ei_max)[0] * self._spectrum_unit * u.keV
        
        event_angles = {'Phi': Phi, 'Psi_sc': Psi, 'Chi_sc': Chi}
        signal_density = get_interp_response(self._dr, event_angles) * integrated_flux * self._exposure_time

        return signal_density

    def _compute_predicted_a(self):
        Eiedges = self._Eiedges
        delta_Em = self._delta_Em
        spectrum = self._likelihood_model.source.spectrum.main.shape
        kernel_name = self._energy_redistribution_kernel
        background_norm = self._bgcounts #self._nuisance_parameters['testing']

        # Mask samples outside the region that will predominantly 
        # contribute to line emission. Eiedges is chosen by user. 
        mask = (self._energy_samples >= Eiedges[0]) & (self._energy_samples <= Eiedges[-1])
        energy_samples = self._energy_samples[mask]
        phi_samples = self._phi_samples[mask]
        psi_samples = self._psi_samples[mask]
        chi_samples = self._chi_samples[mask]

        # Kernal dictionary
        available_kernels = {
                    'gaus': lambda Ei, Em: spectrum(Ei) * self.gaus_kernel(Ei * u.keV, Em, self._sigma_e) * delta_Em,
                    'expgaus': lambda Ei, Em: spectrum(Ei) * self.expgaus_kernel(Ei * u.keV, Em, self._sigma_e, 
                                                                                 self._tau_e) * delta_Em,
                    'doubleexpgaus': lambda Ei, Em: spectrum(Ei) * self.double_expgaus_kernel(Ei * u.keV, Em, self._sigma_e, 
                                                                                 self._w1, self._tau_1, self._tau_2) * delta_Em
                }
        
        redistribution_kernel = available_kernels.get(kernel_name, None)

        # Integration bounds
        Ei_min = Eiedges[0].value
        Ei_max = Eiedges[-1].value

        args = zip(energy_samples, phi_samples, psi_samples, chi_samples)
        compute_func = lambda Em, Phi, Psi, Chi: self._compute_signal_density(Em, Phi, Psi, Chi, Ei_min=Ei_min, Ei_max=Ei_max,
                                                                              redistribution_kernel=redistribution_kernel, spectrum=spectrum)
        
        # signal_densities = []
        # for Em, Phi, Psi, Chi in args:
        #     signal_density = compute_func(Em, Phi, Psi, Chi)
        #     signal_densities.append(signal_density)
        with mp.Pool(processes = 10) as pool:
            signal_densities = Quantity(pool.starmap(compute_func, args))                             # Each entry is the predicted density for one observed event

        background_densities = self._background_kde([energy_samples.value, phi_samples.value, chi_samples.value])        # Currently using the two KDE variables with maximal discriminating power
        eventwise_expectation_densities = Quantity(signal_densities) + background_densities * background_norm

        return eventwise_expectation_densities


    def _compute_predicted_c(self):
        # Compute total expected number of counts over full observation and ROI
        Eiedges = self._Eiedges
        spectrum = self._likelihood_model.source.spectrum.main.shape

        # Kernel dictionary
        available_kernels = {
            'gaus': lambda Ei: spectrum(Ei) * self.integrated_gaus_kernel(Ei * u.keV, Eiedges, self._sigma_e),
            'expgaus': lambda Ei: spectrum(Ei) * self.integrated_expgaus_kernel(Ei * u.keV, Eiedges, self._sigma_e, self._tau_e),
            'doubleexpgaus': lambda Ei: spectrum(Ei) * self.integrated_double_expgaus_kernel(
                Ei * u.keV, Eiedges, self._sigma_e, self._w1, self._tau_1, self._tau_2)
        }

        # Select folded_spectrum function
        folded_spectrum = available_kernels.get(self._energy_redistribution_kernel, spectrum)

        integrated_flux_over_roi = integrate.quad(folded_spectrum, Eiedges[0].value, Eiedges[-1].value)[0] * \
                                   self._spectrum_unit * u.keV

        signal_counts = self._effective_area * integrated_flux_over_roi * self._exposure_time
        background_counts = self._bgcounts #self._nuisance_parameters['testing']
        total_expected_counts = signal_counts + background_counts
        return total_expected_counts

    def get_log_like(self):
        # Final log-likelihood: Poisson likelihood from eventwise 
        # expectation density and total expected counts
        eventwise_expectation_density = self._compute_predicted_a()
        total_expected_counts = self._compute_predicted_c()
        log_like = -total_expected_counts + np.sum(np.log(eventwise_expectation_density + self._tiny))
        return log_like

    def inner_fit(self):
        # Wrapper method for fit logic
        return self.get_log_like()

    def get_number_of_data_points(self):
        return self._n_events

    def get_number_of_model_parameters(self):
        return self._likelihood_model.get_number_of_free_parameters()
    
    def display_model(self, Ei = None, norm = 1, savefig=None):
        # Does not fold with phi and psichi response
        # Incident energy grid (Ei): where photons originate
        if Ei is None:
            Ei = np.linspace(self._Eiedges[0], self._Eiedges[-1], 501)  # keV

        # Measured energy bins (Em): what the detector sees
        Em_bins = np.linspace(self._Eiedges[0], self._Eiedges[-1], 101)  # Measured energy bin edges
        Em_centers = 0.5 * (Em_bins[1:] + Em_bins[:-1])

        # Build redistribution matrix R[Em, Ei]
        if self._energy_redistribution_kernel == 'gaus':
            R = np.array([
                self.gaus_kernel(Ei, Em, self._sigma_e)
                for Em in Em_centers
            ])
        elif self._energy_redistribution_kernel == 'expgaus':
            R = np.array([
                self.expgaus_kernel(Ei, Em, self._sigma_e, self._tau_e)
                for Em in Em_centers
            ])
        elif self._energy_redistribution_kernel == 'doubleexpgaus':
            R = np.array([
                self.double_expgaus_kernel(Ei, Em, self._sigma_e, self._w1, self._tau_1, self._tau_2)
                for Em in Em_centers
            ])

        # Convolve model spectrum with redistribution matrix
        spectrum = self._likelihood_model.source.spectrum.main.shape
        dEi = Ei[1] - Ei[0]  # Energy bin width
        folded_counts = R @ spectrum(Ei - 0.5 * Ei.unit) * dEi  # Expected counts per measured bin

        plt.figure(figsize=(8, 5))
        plt.plot(Em_centers, folded_counts * self._effective_area * self._exposure_time, label="FF", color='darkorange')
        plt.xlabel("Measured Energy (keV)")
        plt.ylabel("Expected Counts")
        plt.title("Forward Folded Model Spectrum")
        # plt.axvline(1149.35)
        counts, _ = np.histogram(self._energy_samples, Em_bins)
        plt.stairs(norm * counts, Em_bins.value, edgecolor='b', lw=2, label='Data')

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if savefig is not None:
            plt.savefig(savefig)
        # plt.show()
        plt.clf()
        # return plt

# %% 
# --- Reset Spectrum Parameters Based on Source Name ---
def spectrum_reset(spec, name='CasAfullyresolved'):
    if name == 'CasAsymmetric':
        # Single Gaussian
        F = 3e-4 / u.cm / u.cm / u.s
        mu = 1157 * u.keV
        sigma = 3.85 / 2.355 * u.keV
        spec.F.value = F.value
        spec.F.min_value = F.value / 1e3
        spec.F.max_value = F.value * 1e3
        spec.F.unit = F.unit
        spec.mu.value = mu.value
        spec.mu.min_value = mu.value - 15
        spec.mu.max_value = mu.value + 15
        spec.mu.unit = mu.unit
        spec.sigma.value = sigma.value
        spec.sigma.min_value = 0.1
        spec.sigma.max_value = sigma.value + 5
        spec.sigma.unit = sigma.unit
    else:
        # Double Gaussian
        F1 = 1e-4 / u.cm / u.cm / u.s
        mu1 = {'CasAfullyresolved': 1149.35, 'CasApartiallyresolved': 1152.41, 'CasAunresolved': 1153.17}[name] * u.keV
        sigma1 = 3.85 * u.keV / 2.355
        F2 = 2 * F1
        mu2 = {'CasAfullyresolved': 1160.89, 'CasApartiallyresolved': 1159.34, 'CasAunresolved': 1158.95}[name] * u.keV
        sigma2 = sigma1

        for i in [1, 2]:
            getattr(spec, f'F_{i}').value = locals()[f'F{i}'].value
            getattr(spec, f'F_{i}').min_value = locals()[f'F{i}'].value / 1e3
            getattr(spec, f'F_{i}').max_value = locals()[f'F{i}'].value * 1e3
            getattr(spec, f'F_{i}').unit = locals()[f'F{i}'].unit
            getattr(spec, f'mu_{i}').value = locals()[f'mu{i}'].value
            getattr(spec, f'mu_{i}').min_value = locals()[f'mu{i}'].value - 10
            getattr(spec, f'mu_{i}').max_value = locals()[f'mu{i}'].value + 10
            getattr(spec, f'mu_{i}').unit = locals()[f'mu{i}'].unit
            getattr(spec, f'sigma_{i}').value = locals()[f'sigma{i}'].value
            getattr(spec, f'sigma_{i}').min_value = 0.1
            getattr(spec, f'sigma_{i}').max_value = locals()[f'sigma{i}'].value + 5
            getattr(spec, f'sigma_{i}').unit = locals()[f'sigma{i}'].unit
    return spec

# %% 
# --- Plot Spectrum Model Before Fitting ---
def plot_spectrum_model(spectrum, spectrum_unit, energy_range=(1140, 1180), num_points=121):
    xdata = np.linspace(energy_range[0], energy_range[1], num_points) * u.keV
    ydata = spectrum(xdata.value) * spectrum_unit

    plt.plot(xdata.value, ydata)
    plt.xlabel(f'Energy ({xdata.unit})')
    plt.ylabel(f'Flux ({spectrum_unit * u.keV})')
    # plt.ylim([0, 1e-5])  # Uncomment if needed
    plt.xscale('log')
    plt.title("Injected Spectrum Model")
    plt.tight_layout()
    plt.show()

# %% 
# --- Build Spectrum Based on Source Name ---
def build_spectrum(name):
    if name == 'CasAsymmetric':
        spectrum = Gaussian()
        spectrum = spectrum_reset(spectrum, name=name)
        spectrum_unit = spectrum.F.unit / spectrum.sigma.unit
        spectrum.F.free = False
        spectrum.mu.free = True
        spectrum.sigma.free = False
    else:
        gaussian1 = Gaussian()
        gaussian2 = Gaussian()
        spectrum = gaussian1 + gaussian2
        spectrum = spectrum_reset(spectrum, name=name)
        spectrum_unit = spectrum.F_1.unit / spectrum.sigma_1.unit
        spectrum.F_1.free = True
        spectrum.mu_1.free = True
        spectrum.sigma_1.free = False
        spectrum.F_2.free = False
        spectrum.mu_2.free = False
        spectrum.sigma_2.free = False
    return spectrum, spectrum_unit

# %% 
# --- Create COSILikeNew Plugin ---
def build_plugin(name, dr2, exposure_time, l, b, energy_samples, phi_samples, Psi_sc_onaxis, Chi_sc_onaxis, spectrum, spectrum_unit, background_kde, bgcounts):
    cosi = COSILikeNew(
        name=name,
        dr=dr2,
        exposure_time=exposure_time,
        energy_samples=energy_samples,
        phi_samples=phi_samples,
        psi_samples=Psi_sc_onaxis,
        chi_samples=Chi_sc_onaxis,
        spectrum_unit=spectrum_unit,
        Eiedges=np.array([1130, 1200]) * u.keV,
        sigma_e=2.41 * u.keV,
        tau_e=1 * u.keV,
        nuisance_param={},
        energy_redistribution_kernel='doubleexpgaus',
        background_kde=background_kde,
        bgcounts=bgcounts
    )

    source = PointSource('source', l=l, b=b, spectral_shape=spectrum)
    model = Model(source)
    cosi.set_model(model)
    return cosi

# %% 
# --- Run Likelihood Fit ---
def run_likelihood(model, plugin):
    plugins = DataList(plugin)
    like = JointLikelihood(model, plugins, verbose=False)
    like.fit()
    return like.results

# %% 
# --- Plot Flux Results ---
def plot_flux_results(results, spectrum, spectrum_unit, name):
    energy = np.linspace(1140 * u.keV, 1180 * u.keV, 81).to_value(u.keV)
    flux_lo, flux_median, flux_hi, flux_inj = np.zeros_like(energy), np.zeros_like(energy), np.zeros_like(energy), np.zeros_like(energy)

    parameters = {
        par.name: results.get_variates(par.path)
        for par in results.optimized_model["source"].parameters.values()
        if par.free
    }

    results_err = results.propagate(results.optimized_model["source"].spectrum.main.shape.evaluate_at, **parameters)

    for i, e in enumerate(energy):
        flux = results_err(e)
        flux_median[i] = flux.median
        flux_lo[i], flux_hi[i] = flux.equal_tail_interval(cl=0.68)
        spectrum_inj = spectrum_reset(spec=spectrum, name=name)
        flux_inj[i] = spectrum_inj.evaluate_at(e)

    fig, ax = plt.subplots()
    plt.plot(energy, energy * energy * flux_median, label="Best fit")
    plt.fill_between(energy, energy * energy * flux_lo, energy * energy * flux_hi, alpha=.5, label="Best fit (errors)")
    plt.plot(energy, energy * energy * flux_inj, color='black', ls=":", label="Injected")
    plt.xscale("log")
    plt.xlabel("Energy (keV)")
    plt.ylabel(r"$E^2 \frac{dN}{dE}$ (keV cm$^{-2}$ s$^{-1}$)")
    plt.legend()
    plt.show()

# %%
def scan_log_likelihood(cosi, spectrum_attr1, values1, spectrum_attr2=None, values2=None):
    logL_results = []

    if spectrum_attr2 is None:
        # One-dimensional scan
        for val1 in values1:
            setattr(cosi._likelihood_model.source.spectrum.main.shape, spectrum_attr1, val1)
            logL = cosi.get_log_like()
            logL_results.append(logL)

        setattr(cosi._likelihood_model.source.spectrum.main.shape, spectrum_attr1, values1[np.argmax(logL_results)])
        return np.array(logL_results)

    else:
        # Two-dimensional scan
        logL_grid = np.zeros((len(values1), len(values2)))
        for i, val1 in enumerate(values1):
            setattr(cosi._likelihood_model.source.spectrum.main.shape, spectrum_attr1, val1)
            for j, val2 in enumerate(values2):
                setattr(cosi._likelihood_model.source.spectrum.main.shape, spectrum_attr2, val2)
                logL_grid[i, j] = cosi.get_log_like()

        idx = np.unravel_index(logL_grid.argmax(), logL_grid.shape)
        setattr(cosi._likelihood_model.source.spectrum.main.shape, spectrum_attr1, values1[idx[0]])
        setattr(cosi._likelihood_model.source.spectrum.main.shape, spectrum_attr2, values2[idx[1]])
        return logL_grid

def plot_logL_1d(values, logLs, xlabel, savefig=None, injected_value=None):
    plt.figure(figsize=(8, 5))
    plt.plot(values, logLs, lw=2)
    if injected_value is not None:
        plt.axvline(injected_value, ls=':', c='g', label='Injected')
    plt.axvline(values[np.nanargmax(logLs)], ls='--', label='Fit')
    
    if (np.log10(values[-1]) - np.log10(values[0])) > 1:      # If range greater than 10x
        plt.xscale('log')
    plt.xlabel(xlabel)
    plt.ylabel("Log-Likelihood")
    plt.title(f"Log-Likelihood vs. {xlabel}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    # plt.show()
    plt.clf()

def plot_logL_2d(values1, values2, logL_grid, xlabel, ylabel, savefig=None):
    plt.figure(figsize=(8, 6))
    plt.contourf(values1, values2, logL_grid.T, levels=50, cmap='viridis')
    plt.colorbar(label="Log-Likelihood")
    
    if (np.log10(values1[-1]) - np.log10(values1[0])) > 1:      # If range greater than 10x
        plt.xscale('log')
    if (np.log10(values2[-1]) - np.log10(values2[0])) > 1:
        plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Log-Likelihood vs. {xlabel} and {ylabel}")
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    # plt.show()
    plt.clf()

# %% 
# --- Initialize Environment ---
def initialize_env(response_path, pix=0):
    configure_plot_style()
    dr = load_detector_response(response_path=response_path, pix=pix)
    dr2 = project_detector_response(dr=dr)
    return dr2

# %% 
# --- Load Signal Data and Coordinates ---
def load_signal_data(DATA_DIR, srcname):
    data, l, b = load_data(DATA_DIR=DATA_DIR, name=srcname)
    rot_custom = hp.Rotator(rot=[l, b - 90], inv=False)
    return data, l, b, rot_custom

# %% 
# --- Load Background and Estimate KDE ---
def load_background_data(DATA_DIR, rot_custom):
    bg_dict = load_background(DATA_DIR=DATA_DIR)
    bg_kde = estimate_kde(bg_dict=bg_dict)
    background_kde, background_data = compute_background_kde(bg_dict=bg_dict, rot_custom=rot_custom)
    return bg_dict, bg_kde, background_kde, background_data

# %% 
# --- Sample Signal and Background Events ---
def create_event_data(data, bg_dict, num_samples, bgcounts, l, b):
    energy_samples, phi_samples, psi_gal_samples, chi_gal_samples = sample_events(
        data=data, bg_dict=bg_dict, num_samples=num_samples, bgcounts=bgcounts)
    # plot_energy_histogram(energy_samples, data)
    Psi_sc_onaxis, Chi_sc_onaxis, pixels_rot, nside = analyze_healpix_coordinates(
        chi_gal_samples=chi_gal_samples, psi_gal_samples=psi_gal_samples,
        l=l, b=b, nside=32)
    # plot_healpix_map(pixels_rot=pixels_rot, nside=nside)
    return energy_samples, phi_samples, Psi_sc_onaxis, Chi_sc_onaxis

# %%
def main2(srcname, num_samples, bgcounts):

    # Init
    dr2 = initialize_env(response_path, pix=0)

    # Signal
    data, l, b, rot_custom = load_signal_data(DATA_DIR=DATA_DIR, srcname=srcname)

    # Background
    bg_dict, bg_kde, background_kde, background_data = load_background_data(DATA_DIR=DATA_DIR, rot_custom=rot_custom)

    # All events
    energy_samples, phi_samples, Psi_sc_onaxis, Chi_sc_onaxis = create_event_data(
        data=data, bg_dict=bg_dict, num_samples=num_samples, bgcounts=bgcounts, l=l, b=b)

    # Create spectrum and threeML plugin
    spectrum, spectrum_unit = build_spectrum(srcname)       # Change to model you want to fit (can be different from data)
    # plot_spectrum_model(spectrum=spectrum, spectrum_unit=spectrum_unit, energy_range=(1140, 1180), num_points=121)
    exposure_time = 92.34 * u.d * (num_samples / len(data['Energies'])) * 0.35
    print(len(np.where((energy_samples.value > 1130) & (energy_samples.value < 1200))[0]),
          len(np.where((data['Energies'] > 1130) & (data['Energies'] < 1200))[0]),
          len(data['Energies']))

    cosi = build_plugin('cosi', dr2, exposure_time, l, b, energy_samples, phi_samples, 
                         Psi_sc_onaxis, Chi_sc_onaxis, spectrum, spectrum_unit, background_kde, bgcounts)

    return cosi

# %% 
# --- Main Execution ---
def main():

    sha = get_git_revision_short_hash()
    srcname = 'CasAfullyresolved'
    num_samples = 1000

    bgcounts = 0
    for bgcounts in [0, 400, 800]:
        cosi = main2(srcname, num_samples, bgcounts)
        spectrum = cosi._likelihood_model.source.spectrum.main.shape
        # spectrum_unit = spectrum.F.unit / spectrum.sigma.unit 
        model = cosi._likelihood_model

        # results = run_likelihood(model, cosi)
        # print(results.display())
        # print(results.optimized_model["source"])
        # plot_flux_results(results, spectrum, spectrum_unit, name=srcname)

        # # Log-likelihood scan
        counter = 0
        while True:
            savefig = f'N{srcname}_S{num_samples}_B{bgcounts}_#{sha}_{counter}.png'
            if not os.path.exists('FF/' + savefig):
                break
            counter += 1
        F_values = np.geomspace(1e-5, 1e-2, 15)
        mu_values = np.linspace(1145, 1155, 5)
        sigma_values = np.linspace(1.1, 2.15, 7)
        logL_grid = scan_log_likelihood(cosi, 'F_1', F_values)
        plot_logL_1d(F_values, logL_grid, xlabel='F_1', savefig='logL/' + savefig)
        # logL_grid = scan_log_likelihood(cosi, 'mu', mu_values, 'sigma', sigma_values)
        # plot_logL_2d(mu_values, sigma_values, logL_grid, xlabel='mu', ylabel='sigma', savefig='logL/' + savefig)

        cosi.display_model(savefig='FF/' + savefig)


# %% 
# --- Script Entry Point ---
# Simply run python LMDR4.py within appropriate environment
if __name__ == "__main__":
    main()
