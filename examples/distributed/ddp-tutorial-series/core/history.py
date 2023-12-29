from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, OrderedDict, TypeVar, Union

import torch
from core.state import EpochState

# H = TypeVar('H', bound='History')


# @dataclass
# class History:
#     batch_states: List[State] = field(default_factory=list)
#     epoch_states: List[State] = field(default_factory=list)

#     def add_batch_state(self, state: State) -> None:
#         """Add a new state after processing a batch."""
#         self.batch_states.append(state)

#     def add_epoch_state(self, state: State) -> None:
#         """Add a new state after processing an epoch."""
#         self.epoch_states.append(state)

#     def get_batch_state(self, batch_index: int) -> State:
#         """Retrieve a state from a specific batch index."""
#         return self.batch_states[batch_index]

#     def get_epoch_state(self, epoch_index: int) -> State:
#         """Retrieve a state from a specific epoch index."""
#         return self.epoch_states[epoch_index]


@dataclass
class History:
    states: List[EpochState] = field(default_factory=list)

    def add_state(self, state: EpochState) -> None:
        """Add a new state after processing a batch."""
        copied_state = deepcopy(state)
        self.states.append(copied_state)
