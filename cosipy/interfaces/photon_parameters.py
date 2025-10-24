from typing import Protocol, runtime_checkable

from astropy import units as u
from astropy.coordinates import BaseCoordinateFrame, SkyCoord
from scoords import SpacecraftFrame


@runtime_checkable
class PhotonInterface(Protocol):
    """
    Derived classes have all access methods
    """

@runtime_checkable
class PhotonWithEnergyInterface(PhotonInterface, Protocol):

    @property
    def energy_keV(self) -> float:...

    @property
    def energy(self) -> u.Quantity:
        """
        Add fancy energy quantity
        """
        return u.Quantity(self.energy_keV, u.keV)

@runtime_checkable
class PhotonWithDirectionInterface(PhotonInterface, Protocol):

    @property
    def frame(self) -> BaseCoordinateFrame:...

    @property
    def direction_lon_radians(self) -> float: ...

    @property
    def direction_lat_radians(self) -> float: ...

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
class PhotonInSCFrameInterface(PhotonInterface, Protocol):

    @property
    def frame(self) -> SpacecraftFrame:...

class PhotonWithDirectionInSCFrameInterface(PhotonWithDirectionInterface,
                                            PhotonInSCFrameInterface):
    pass

class PhotonWithDirectionAndEnergyInSCFrameInterface(PhotonWithDirectionInSCFrameInterface,
                                                     PhotonWithEnergyInterface):
    pass


