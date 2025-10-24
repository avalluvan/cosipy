import itertools
from typing import Union, Iterable

import numpy as np
from astropy.time import Time

from cosipy.interfaces import TimeTagEventInterface, EventInterface
from cosipy.interfaces.event_selection import EventSelectorInterface
from cosipy.util.iterables import itertools_batched


class MultiTimeSelector(EventSelectorInterface):

    def __init__(self, tstart_list:Time = None, tstop_list:Time = None, batch_size:int = 10000):
        """
        Assumes events are time-ordered

        Parameters
        ----------
        tstart_list:
        tstop_list:
        batch_size:
        """
        if tstart_list.isscalar == True:
            tstart_list = Time([tstart_list])
        if tstop_list.isscalar == True:
            tstop_list = Time([tstop_list])

        self._tstart_list = tstart_list
        self._tstop_list = tstop_list

        self._batch_size = batch_size
    
    @classmethod
    def load_GTI(cls, gti, batch_size:int = 10000):
        """
        Instantiate a multi time selector from a good time intervals.

        Parameters
        ----------
        gti:
        batch_size:
        """
        tstart_list = gti.tstart_list
        tstop_list = gti.tstop_list

        selector = cls(tstart_list, tstop_list, batch_size)

        return selector

    def _select(self, event:TimeTagEventInterface) -> bool:
        # Single event
        return next(iter(self.select([event])))

    def select(self, events:Union[TimeTagEventInterface, Iterable[TimeTagEventInterface]]) -> Union[bool, Iterable[bool]]:

        if isinstance(events, EventInterface):
            # Single event
            return self._select(events)
        else:
            # Multiple

            # Working in chunks/batches.
            # This can optimized based on the system

            for chunk in itertools_batched(events, self._batch_size):

                jd1 = []
                jd2 = []

                for event in chunk:
                    jd1.append(event.jd1)
                    jd2.append(event.jd2)

                time = Time(jd1, jd2, format = 'jd')

                indices = np.searchsorted(self._tstart_list, time, side='right') - 1
                valid = (indices >= 0) & (indices < len(self._tstop_list))
                result = np.zeros(len(time), dtype=bool)
                result[valid] = time[valid] <= self._tstop_list[indices[valid]]

                for sel in result:
                    yield sel
