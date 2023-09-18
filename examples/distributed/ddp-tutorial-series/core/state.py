from dataclasses import asdict, dataclass
from typing import Any, Dict, List, OrderedDict, Union

import torch

from core.serializable import Serializable


@dataclass
class State(Serializable):
    model_state: OrderedDict[str, torch.Tensor]
    optimizer_state: Dict[str, Any]
    scheduler_state: Dict[str, Any]
    torch_rng_state: torch.ByteTensor
    epoch_index: int
    batch_index: int
    lr_or_ls_this_epoch: Union[float, List[float]]
    avg_train_loss_per_sample_this_epoch: float
    avg_valid_loss_per_sample_this_epoch: float
    avg_train_loss_per_sample_this_batch: float
    avg_valid_loss_per_sample_this_batch: float

    def state_dict(self) -> Dict[str, Any]:
        """Convert the State dataclass to a dictionary."""
        return asdict(self)

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load the state dictionary into the dataclass."""
        self.__dict__.update(state)
