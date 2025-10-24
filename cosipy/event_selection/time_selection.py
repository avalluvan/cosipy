import itertools
from typing import Union, Iterable

import numpy as np
from astropy.time import Time

from cosipy.interfaces import TimeTagEventInterface, EventInterface
from cosipy.interfaces.event_selection import EventSelectorInterface
from cosipy.util.iterables import itertools_batched


class TimeSelector(EventSelectorInterface):

    def __init__(self, tstart:Time = None, tstop:Time = None, batch_size:int = 10000):
        """
        Assumes events are time-ordered

        Parameters
        ----------
        chunk_size : object
            Number of events processed at a time
        tstart
        tstop
        """

        self._tstart = tstart
        self._tstop = tstop

        self._batch_size = batch_size

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

                selected = np.logical_and(np.logical_or(self._tstart is None, time > self._tstart),
                                          np.logical_or(self._tstop is None,  time <= self._tstop))

                for sel in selected:
                    yield sel

                if self._tstop is not None and time[-1] > self._tstop:
                    # Stop further loading of event
                    return
