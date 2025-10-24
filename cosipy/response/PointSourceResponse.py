from histpy import Histogram
from astropy.coordinates import SkyCoord
from astropy.units import Quantity

from cosipy.polarization.polarization_axis import PolarizationAxis
from cosipy.threeml.util import to_linear_polarization
from mhealpy import HealpixMap
from cosipy.interfaces import BinnedInstrumentResponseInterface, BinnedDataInterface
from histpy import Histogram, Axis, Axes  # , Axes, Axis

import numpy as np
import astropy.units as u
from scoords import Attitude

from .functions import get_integrated_spectral_model

import logging

from cosipy.spacecraftfile import SpacecraftAttitudeMap
from ..data_io import EmCDSBinnedData

logger = logging.getLogger(__name__)

class PointSourceResponse(Histogram):
    """
    Handles the multi-dimensional matrix that describes the expected
    response of the instrument for a particular point in the sky.

    Parameters
    ----------
    axes : :py:class:`histpy.Axes`
        Binning information for each variable. The following labels are expected:\n
        - ``Ei``: Real energy
        - ``Em``: Measured energy. Optional
        - ``Phi``: Compton angle. Optional.
        - ``PsiChi``:  Location in the Compton Data Space (HEALPix pixel). Optional.
        - ``SigmaTau``: Electron recoil angle (HEALPix pixel). Optional.
        - ``Dist``: Distance from first interaction. Optional.
    contents : array, :py:class:`astropy.units.Quantity` or :py:class:`sparse.SparseArray`
        Array containing the differential effective area convolved with wht source exposure.
    unit : :py:class:`astropy.units.Unit`, optional
        Physical units, if not specified as part of ``contents``. Units of ``area*time``
        are expected.
    """
    
    @property
    def photon_energy_axis(self):
        """
        Real energy bins (``Ei``).

        Returns
        -------
        :py:class:`histpy.Axes`
        """
        
        return self.axes['Ei']

    @property
    def measurement_axes(self):
        return self.axes['Em', 'Phi', 'PsiChi']

    def get_expectation(self, spectrum, polarization=None):
        """
        Convolve the response with a spectral (and optionally, polarization) hypothesis to obtain the expected
        excess counts from the source.

        Parameters
        ----------
        spectrum : :py:class:`threeML.Model`
            Spectral hypothesis.
        polarization : 'astromodels.core.polarization.LinearPolarization', optional
            Polarization angle and degree. The angle is assumed to have same convention as point source response.
        
        Returns
        -------
        :py:class:`histpy.Histogram`
             Histogram with the expected counts on each analysis bin
        """

        polarization = to_linear_polarization(polarization)

        if polarization is None:

            if 'Pol' in self.axes.labels:

                raise RuntimeError("Must include polarization in point source response if using polarization response")

            contents = self.contents

        else:

            if not 'Pol' in self.axes.labels:
                
                raise RuntimeError("Response must have polarization angle axis to include polarization in point source response")

            polarization_angle = polarization.angle.value
            polarization_level = polarization.degree.value / 100.

            if polarization_angle == 180.:
                polarization_angle = 0.

            unpolarized_weights = np.full(self.axes['Pol'].nbins, (1. - polarization_level) / self.axes['Pol'].nbins)
            polarized_weights = np.zeros(self.axes['Pol'].nbins)

            polarization_bin_index = self.axes['Pol'].find_bin(polarization_angle * u.deg)
            polarized_weights[polarization_bin_index] = polarization_level

            weights = unpolarized_weights + polarized_weights

            contents = np.tensordot(weights, self.contents, axes=([0], [self.axes.label_to_index('Pol')]))


        energy_axis = self.photon_energy_axis

        flux = get_integrated_spectral_model(spectrum, energy_axis)
        
        expectation = np.tensordot(contents, flux.contents, axes=([0], [0]))

        # if self is sparse, expectation will be a SparseArray with
        # no units, so set the result's unit explicitly
        hist = Histogram(self.measurement_axes, contents = expectation,
                         unit = self.unit * flux.unit,
                         copy_contents = False)

        if not hist.unit == u.dimensionless_unscaled:
            raise RuntimeError("Expectation should be dimensionless, but has units of " + str(hist.unit) + ".")

        return hist

    @classmethod
    def from_dwell_time_map(cls,
                            data:BinnedDataInterface,
                            response: BinnedInstrumentResponseInterface,
                            exposure_map: HealpixMap,
                            energy_axis: Axis,
                            polarization_axis: PolarizationAxis = None
                            ):

        axes = [energy_axis]

        polarization_centers = None
        if polarization_axis is not None:
            axes += [polarization_axis]
            polarization_centers = polarization_axis.centers

        axes += list(data.axes)

        psr = PointSourceResponse(axes, unit=u.cm * u.cm * u.s)

        for p in range(exposure_map.npix):

            coord = exposure_map.pix2skycoord(p)

            if exposure_map[p] != 0:
                psr += response.differential_effective_area(data, coord, energy_axis.centers, polarization_centers) * exposure_map[p]

        return psr

    @classmethod
    def from_scatt_map(cls,
                        coord: SkyCoord,
                        data:BinnedDataInterface,
                        response: BinnedInstrumentResponseInterface,
                        scatt_map: SpacecraftAttitudeMap,
                        energy_axis: Axis,
                        polarization_axis: PolarizationAxis = None
                        ):
        """

        Parameters
        ----------
        measured_axes
        response
        scatt_map
        energy_axis
        polarization_axis

        Returns
        -------

        """

        if not isinstance(data, EmCDSBinnedData):
            raise TypeError(f"Wrong data type '{type(data)}', expected {EmCDSBinnedData}.")

        axes = [energy_axis]

        if polarization_axis is not None:
            axes += [polarization_axis]

        axes += list(data.axes)
        axes = Axes(axes)

        psr = Quantity(np.empty(shape=axes.shape), unit = u.cm * u.cm * u.s)

        for i, (pixels, exposure) in \
                enumerate(zip(scatt_map.contents.coords.transpose(),
                              scatt_map.contents.data * scatt_map.unit)):

            att = Attitude.from_axes(x=scatt_map.axes['x'].pix2skycoord(pixels[0]),
                                     y=scatt_map.axes['y'].pix2skycoord(pixels[1]))


            response.differential_effective_area(data,
                                                 coord,
                                                 energy_axis.centers,
                                                 None if polarization_axis is None else polarization_axis.centers,
                                                 attitude = att,
                                                 weight=exposure,
                                                 out=psr,
                                                 add_inplace=True)

        return PointSourceResponse(axes, contents = psr)

