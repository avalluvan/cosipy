import itertools
from typing import Protocol, ClassVar, Type, Iterator, runtime_checkable, Iterable

from astropy.coordinates import BaseCoordinateFrame, SkyCoord
from scoords import SpacecraftFrame

from .photon_parameters import PhotonInterface, PhotonWithEnergyInterface

import astropy.units as u

@runtime_checkable
class PhotonListInterface(Protocol):

    # Type returned by __iter__
    photon_type = ClassVar[Type]

    def __iter__(self) -> Iterator[PhotonInterface]:
        """
        Return one Event at a time
        """
    def __getitem__(self, item: int) -> PhotonInterface:
        """
        Convenience method. Pretty slow in general. It's suggested that
        the implementations override it
        """
        return next(itertools.islice(self, item, None))

    @property
    def nphotons(self) -> int:
        """
        Total number of events yielded by __iter__

        Convenience method. Pretty slow in general. It's suggested that
        the implementations override it
        """
        return sum(1 for _ in iter(self))

@runtime_checkable
class EventDataWithEnergyInterface(PhotonListInterface, Protocol):

    def __iter__(self) -> Iterator[PhotonWithEnergyInterface]:...

    @property
    def energy_radians(self) -> Iterable[float]:...

    @property
    def energy(self) -> u.Quantity:
        """
        Add fancy energy quantity
        """
        return u.Quantity(self.energy_radians, u.radians)

@runtime_checkable
class PhotonListWithDirectionInterface(PhotonListInterface, Protocol):

    @property
    def frame(self) -> BaseCoordinateFrame:...

    @property
    def direction_lon_radians(self) -> Iterable[float]: ...

    @property
    def direction_lat_radians(self) -> Iterable[float]: ...

    @property
    def direction_direction(self) -> SkyCoord:
        """
        Add fancy energy quantity
        """
        return SkyCoord(self.direction_lon_radians,
                        self.direction_lat_radians,
                        unit=u.rad,
                        frame=self.frame)

@runtime_checkable
class PhotonListInSCFrameInterface(PhotonListInterface, Protocol):

    @property
    def frame(self) -> SpacecraftFrame:...



