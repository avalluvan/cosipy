from scoords import SpacecraftFrame

from cosipy.interfaces.photon_parameters import PhotonWithDirectionAndEnergyInSCFrameInterface


class PhotonWithDirectionAndEnergyInSCFrame(PhotonWithDirectionAndEnergyInSCFrameInterface):

    frame = SpacecraftFrame()

    def __init__(self, direction_lon_radians, direction_lat_radians, energy_keV):
        self._energy = energy_keV
        self._lon = direction_lon_radians
        self._lat = direction_lat_radians

    @property
    def energy_keV(self) -> float:
        return self._energy

    @property
    def direction_lon_radians(self) -> float:
        return self._lon

    @property
    def direction_lat_radians(self) -> float:
        return self._lat
