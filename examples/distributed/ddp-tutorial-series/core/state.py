from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, OrderedDict, Union

import torch

from core.serializable import Serializable


@dataclass
class State(Serializable):
    model_state: Optional[OrderedDict[str, torch.Tensor]] = None
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    torch_rng_state: Optional[torch.ByteTensor] = None
    epoch_index: int = -1
    batch_index: int = -1
    lr_or_ls_this_epoch: Optional[Union[float, List[float]]] = None
    avg_train_loss_per_sample_this_epoch: float = -1
    avg_valid_loss_per_sample_this_epoch: float = -1
    avg_train_loss_per_sample_this_batch: float = -1
    avg_valid_loss_per_sample_this_batch: float = -1

    def state_dict(self) -> Dict[str, Any]:
        """Convert the State dataclass to a dictionary."""
        return asdict(self)

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load the state dictionary into the dataclass."""
        self.__dict__.update(state)
