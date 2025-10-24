import itertools
from typing import Protocol, runtime_checkable, Dict, Any, Iterator, Sequence, Generator, Iterable, Union, Optional, \
    Tuple

from . import EventInterface

@runtime_checkable
class EventSelectorInterface(Protocol):

    def select(self, event:Union[EventInterface, Iterable[EventInterface]]) -> Union[bool, Iterable[bool]]:
        """
        True to keep an event

        Return a single value for a single Event.
        As many values for an Iterable of events
        """

    def mask(self, events: Iterable[EventInterface]) -> Iterable[Tuple[bool,EventInterface]]:
        """
        Returns an iterable of tuples. Each tuple has 2 elements:
        - First: True to keep an event, False to filter it out.
        - Second: the event itself.
        """
        events1, events2 = itertools.tee(events, 2)
        for selected, event in zip(self.select(events1), events2):
            yield selected, event

    def __call__(self, events: Iterable[EventInterface]) -> Union[Iterable[EventInterface], None]:
        """
        Skips events that were not selected

        Returning None raises StopIteration
        """
        for selected,event in self.mask(events):
            if selected:
                yield event


