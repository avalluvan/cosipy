import numpy as np
from astropy.time import Time
from astropy.io import fits

class GoodTimeInterval():
    
    def __init__(self, tstart_list, tstop_list):
        """
        Initialize GTI object.
        
        Parameters
        ----------
        tstart_list : astropy.time.Time (array)
            Start times of GTI intervals
        tstop_list : astropy.time.Time (array)
            Stop times of GTI intervals
        """
        # Check that starts and stops are scalar
        if tstart_list.isscalar == True:
            tstart_list = Time([tstart_list])
        if tstop_list.isscalar == True:
            tstop_list = Time([tstop_list])
        
        self._tstart_list = tstart_list
        self._tstop_list = tstop_list
        
        # Sort by start time
        self.sort()

    @property
    def tstart_list(self):
        return self._tstart_list
    
    @property
    def tstop_list(self):
        return self._tstop_list
    
    def sort(self):
        """
        Sort GTI by start time in ascending order.
        
        Modifies the GTI in place.
        Stops are sorted according to the start time order.
        """
        sort_idx = np.argsort(self._tstart_list)
        self._tstart_list = self._tstart_list[sort_idx]
        self._tstop_list = self._tstop_list[sort_idx]
