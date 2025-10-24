import itertools
import operator
from typing import Protocol, Union, Optional, Iterable, Tuple, runtime_checkable, ClassVar

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.units import Quantity
from histpy import Axes, Histogram

from astropy import units as u
from scoords import Attitude

from cosipy.interfaces import BinnedDataInterface, ExpectationDensityInterface, BinnedExpectationInterface, EventInterface
from cosipy.interfaces.photon_list import PhotonListWithDirectionInterface
from cosipy.interfaces.photon_parameters import PhotonInterface, PhotonWithDirectionInterface
from cosipy.polarization import PolarizationAngle

__all__ = ["BinnedInstrumentResponseInterface"]

class BinnedInstrumentResponseInterface(BinnedExpectationInterface, Protocol):

    def differential_effective_area(self,
                                    data: BinnedDataInterface,
                                    direction: SkyCoord,
                                    energy:u.Quantity,
                                    polarization:PolarizationAngle,
                                    attitude:Attitude,
                                    weight: Union[Quantity, float],
                                    out: Quantity,
                                    add_inplace: bool) -> Quantity:
        """

        Parameters
        ----------
        data:
            Binned data
        direction:
            Photon incoming direction. If not in a SpacecraftFrame, then provide an attitude for the transformation
        energy:
            Photon energy
        polarization
            Photon polarization angle. If the coordinate frame of the polarization convention is not a
            SpacecraftFrame, then provide an attitude for the transformation
        attitude
            Attitude defining the orientation of the SC in an inertial coordinate system.
        weight
            Optional. Weighting the result by a given weight. Providing the weight at this point as opposed to
            apply it to the output can result in greater efficiency.
        out
            Optional. Histogram to store the output. If possible, the implementation should try to avoid allocating
            new memory.
        add_inplace
            Optional. If True and a Histogram output was provided, the implementation should try to avoid allocating new
            memory and add --not set-- the result of this operation to the output.

        Returns
        -------
        The effective area times the event measurement probability distribution integrated on each of the bins
        of the provided axes. It has the shape (direction.shape, energy.shape, polarization.shape, axes.shape)
        """

@runtime_checkable
class InstrumentResponseFunctionInterface(Protocol):

    # The photon class and event class that the IRF implementation can handle
    photon_type = ClassVar[PhotonInterface]
    event_type = ClassVar[EventInterface]

    def event_probability(self, query: Iterable[Tuple[PhotonInterface, EventInterface]]) -> Iterable[float]:
        """
        Return the probability density of measuring a given event given a photon.

        The units of the output the inverse of the phase space of the class event_type data space.
        e.g. if the event measured energy in keV, the units of output of this function are implicitly 1/keV
        """

    def random_events(self, photons:Iterable[PhotonInterface]) -> Iterable[EventInterface]:
        """
        Return a stream of random events, one per photon
        """

@runtime_checkable
class FarFieldInstrumentResponseFunctionInterface(InstrumentResponseFunctionInterface, Protocol):

    def effective_area_cm2(self, photons: Iterable[PhotonWithDirectionInterface]) -> Iterable[float]:
        """

        """

    def differential_effective_area_cm2(self, query: Iterable[Tuple[PhotonWithDirectionInterface, EventInterface]]) -> Iterable[float]:
        """
        Event probability multiplied by effective area

        This is provided as a helper function assuming the child classes implemented event_probability
        """

        # Guard to avoid infinite recursion in incomplete child classes
        cls = type(self)
        if (cls.differential_effective_area_cm2 is FarFieldInstrumentResponseFunctionInterface.differential_effective_area_cm2
            and
            cls.event_probability is FarFieldInstrumentResponseFunctionInterface.event_probability):
            raise NotImplementedError("Implement differential_effective_area_cm2 and/or event_probability")

        query1, query2 = itertools.tee(query, 2)
        photon_query = [photon for photon,_ in query1]

        return map(operator.mul, self.effective_area_cm2(photon_query), self.event_probability(query2))

    def event_probability(self, query: Iterable[Tuple[PhotonWithDirectionInterface, EventInterface]]) -> Iterable[float]:
        """
        Return the probability density of measuring a given event given a photon.

        In the far field case it is the same as the differential_effective_area_cm2 divided by the effective area

        This is provided as a helper function assuming the child classes implemented differential_effective_area_cm2
        """

        # Guard to avoid infinite recursion in incomplete child classes
        cls = type(self)
        if (
                cls.differential_effective_area_cm2 is FarFieldInstrumentResponseFunctionInterface.differential_effective_area_cm2
                and
                cls.event_probability is FarFieldInstrumentResponseFunctionInterface.event_probability):
            raise NotImplementedError("Implement differential_effective_area_cm2 and/or event_probability")

        query1, query2 = itertools.tee(query, 2)
        photon_query = [photon for photon, _ in query1]

        return map(operator.truediv, self.differential_effective_area_cm2(query2), self.effective_area_cm2(photon_query))


    def effective_area(self, photons: Iterable[PhotonWithDirectionInterface]) -> Iterable[u.Quantity]:
        """
        Convenience function
        """
        for area_cm2 in self.effective_area_cm2(photons):
            yield u.Quantity(area_cm2, u.cm*u.cm)

    def differential_effective_area(self, query: Iterable[Tuple[PhotonWithDirectionInterface, EventInterface]]) -> Iterable[u.Quantity]:
        for area_cm2 in self.differential_effective_area(query):
            yield u.Quantity(area_cm2, u.cm*u.cm)





















