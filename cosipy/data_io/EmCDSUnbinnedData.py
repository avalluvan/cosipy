from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple, Union, List

import numpy as np
from astropy.coordinates import BaseCoordinateFrame, Angle, SkyCoord, UnitSphericalRepresentation
from astropy.time import Time
from astropy.units import Quantity
from numpy._typing import ArrayLike
from scoords import SpacecraftFrame

from cosipy import UnBinnedData
from cosipy.interfaces import EventWithEnergyInterface, EventDataInterface, EventDataWithEnergyInterface
from cosipy.interfaces.data_interface import  TimeTagEmCDSEventDataInSCFrameInterface
from cosipy.interfaces.event import  TimeTagEmCDSEventInSCFrameInterface, \
    EmCDSEventInSCFrameInterface

import astropy.units as u

from cosipy.interfaces.event_selection import EventSelectorInterface

class EmCDSEventInSCFrame(EmCDSEventInSCFrameInterface):

    _frame = SpacecraftFrame()

    def __init__(self, energy, scatt_angle, scatt_lon, scatt_lat, event_id = None):
        """
        Parameters
        ----------
        jd1: julian days
        jd2: julian days
        energy: keV
        scatt_angle: scattering angle radians
        scatt_lon: scattering longitude radians
        scatt_lat: scattering latitude radians
        """
        self._id = event_id
        self._energy = energy
        self._scatt_angle = scatt_angle
        self._scatt_lat = scatt_lat
        self._scatt_lon = scatt_lon

    @property
    def id(self) -> int:
        return self._id

    @property
    def frame(self):
        return self._frame

    @property
    def energy_keV(self) -> float:
        return self._energy

    @property
    def scattering_angle_rad(self) -> float:
        return self._scatt_angle

    @property
    def scattered_lon_rad_sc(self) -> float:
        return self._scatt_lon

    @property
    def scattered_lat_rad_sc(self) -> float:
        return self._scatt_lat

class TimeTagEmCDSEventInSCFrame(EmCDSEventInSCFrame, TimeTagEmCDSEventInSCFrameInterface):

    def __init__(self, jd1, jd2, energy, scatt_angle, scatt_lon, scatt_lat, event_id=None):
        """
        Parameters
        ----------
        jd1: julian days
        jd2: julian days
        energy: keV
        scatt_angle: scattering angle radians
        scatt_lon: scattering longitude radians
        scatt_lat: scattering latitude radians
        """
        super().__init__(energy, scatt_angle, scatt_lon, scatt_lat, event_id)

        self._jd1 = jd1
        self._jd2 = jd2

    @property
    def jd1(self):
        return self._jd1

    @property
    def jd2(self):
        return self._jd2



class TimeTagEmCDSEventDataInSCFrameFromArrays(TimeTagEmCDSEventDataInSCFrameInterface):
    """

    """

    _frame = SpacecraftFrame()
    event_type = TimeTagEmCDSEventInSCFrame

    def __init__(self,
                 time:Time,
                 energy:Quantity,
                 scattering_angle:Angle,
                 scattered_direction:SkyCoord,
                 event_id:Optional[Iterable[int]] = None,
                 selection:EventSelectorInterface = None):
        """

        Parameters
        ----------
        time
        energy
        scattering_angle
        scattered_direction
        event_id
        selection
        """

        self._jd1 = time.jd1
        self._jd2 = time.jd2
        self._energy = energy.to_value(u.keV)
        self._scatt_angle = scattering_angle.to_value(u.rad)

        if not isinstance(scattered_direction.frame, SpacecraftFrame):
            raise ValueError("Coordinates need to be in SC frame")

        scattered_direction = scattered_direction.represent_as(UnitSphericalRepresentation)

        self._scatt_lat = scattered_direction.lat.rad
        self._scatt_lon = scattered_direction.lon.rad
        if event_id is None:
            self._id = np.arange(self._jd1.size)
        else:
            self._id = np.asarray(event_id)

        # Check size
        self._id, self._jd1, self._jd2, self._energy, self._scatt_angle, self._scatt_lat, self._scatt_lon = np.broadcast_arrays(self._id, self._jd1, self._jd2, self._energy, self._scatt_angle, self._scatt_lat, self._scatt_lon)

        self._nevents = self._id.size

        if selection is not None:
            # Apply selection once and for all
            new_id = []
            new_jd1 = []
            new_jd2 = []
            new_energy = []
            new_scatt_angle = []
            new_scatt_lat = []
            new_scatt_lon = []

            nevents = 0
            for event in selection(self):
                new_id.append(event.id)
                new_jd1.append(event.jd1)
                new_jd2.append(event.jd2)
                new_energy.append(event.energy_keV)
                new_scatt_angle.append(event.scattering_angle_rad)
                new_scatt_lat.append(event.scattered_lat_rad_sc)
                new_scatt_lon.append(event.scattered_lon_rad_sc)
                nevents +=  1

            self._nevents = nevents

            self._id = np.asarray(new_id)
            self._jd1 = np.asarray(new_jd1)
            self._jd2 = np.asarray(new_jd2)
            self._energy = np.asarray(new_energy)
            self._scatt_angle = np.asarray(new_scatt_angle)
            self._scatt_lat = np.asarray(new_scatt_lat)
            self._scatt_lon = np.asarray(new_scatt_lon)

    def __getitem__(self, i: int) -> TimeTagEmCDSEventInSCFrameInterface:
        return TimeTagEmCDSEventInSCFrame(self._jd1[i], self._jd2[i], self._energy[i], self._scatt_angle[i], self._scatt_lon[i], self._scatt_lat[i],
                                          self._id[i])

    @property
    def nevents(self) -> int:
        return self._nevents

    def __iter__(self) -> Iterator[TimeTagEmCDSEventInSCFrameInterface]:
        for id, jd1, jd2, energy, scatt_angle, scatt_lat, scatt_lon in zip(self._id, self._jd1, self._jd2, self._energy, self._scatt_angle, self._scatt_lat, self._scatt_lon):
            yield TimeTagEmCDSEventInSCFrame(jd1, jd2, energy, scatt_angle, scatt_lon, scatt_lat, id)

    @property
    def frame(self) -> SpacecraftFrame:
        return self._frame

    @property
    def ids(self) -> Iterable[int]:
        return self._id

    @property
    def jd1(self) -> Iterable[float]:
        return self._jd1

    @property
    def jd2(self) -> Iterable[float]:
        return self._jd2

    @property
    def energy_keV(self) -> Iterable[float]:
        return self._energy

    @property
    def scattering_angle_rad(self) -> Iterable[float]:
        return self._scatt_angle

    @property
    def scattered_lon_rad_sc(self) -> Iterable[float]:
        return self._scatt_lon

    @property
    def scattered_lat_rad_sc(self) -> Iterable[float]:
        return self._scatt_lat

class TimeTagEmCDSEventDataInSCFrameFromDC3Fits(TimeTagEmCDSEventDataInSCFrameFromArrays):

    def __init__(self, data_path: Union[Path, List[Path]],
                 selection:EventSelectorInterface = None):

        time = np.empty(0)
        energy = np.empty(0)
        phi = np.empty(0)
        psi = np.empty(0)
        chi = np.empty(0)

        if isinstance(data_path, (str, Path)):
            data_path = [Path(data_path)]

        for file in data_path:
            # get_dict_from_fits is really a static method, no config file needed
            data_dict = UnBinnedData.get_dict_from_fits(None, str(file))

            time = np.append(time, data_dict['TimeTags'])
            energy = np.append(energy, data_dict['Energies'])
            phi = np.append(phi, data_dict['Phi'])
            psi = np.append(psi, data_dict['Psi local'])
            chi = np.append(chi, data_dict['Chi local'])

        # Time sort
        tsort = np.argsort(time)

        time = time[tsort]
        energy = energy[tsort]
        phi = phi[tsort]
        psi = psi[tsort]
        chi = chi[tsort]

        time = Time(time, format='unix')
        energy = u.Quantity(energy, u.keV)
        phi = Angle(phi, u.rad)
        # Psi is colatitude (latitude complementary angle)
        psichi = SkyCoord(chi, np.pi / 2 - psi, unit=u.rad,
                          frame=SpacecraftFrame())

        super().__init__(time, energy, phi, psichi, selection = selection)




