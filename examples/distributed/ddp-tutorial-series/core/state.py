from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, OrderedDict, Union

import torch

from core.serializable import Serializable


@dataclass
class State(Serializable):
    model_state: Optional[OrderedDict[str, torch.Tensor]] = field(
        default=None, metadata={"help": "State dictionary of the model."}
    )
    optimizer_state: Optional[Dict[str, Any]] = field(
        default=None, metadata={"help": "State dictionary of the optimizer."}
    )
    scheduler_state: Optional[Dict[str, Any]] = field(
        default=None, metadata={"help": "State dictionary of the scheduler."}
    )
    torch_rng_state: Optional[torch.ByteTensor] = field(
        default=None, metadata={"help": "Random number generator state of PyTorch."}
    )
    epoch_index: int = field(
        default=-1,
        metadata={
            "help": "Current epoch index. Initial value is -1 to indicate it's not started yet."
        },
    )
    batch_index: int = field(
        default=-1,
        metadata={
            "help": "Current batch index within the current epoch. Initial value is -1 to indicate it's not started yet."
        },
    )
    lr_or_ls_this_epoch: Optional[Union[float, List[float]]] = field(
        default=None,
        metadata={"help": "Learning rate or learning schedule for the current epoch."},
    )
    avg_train_loss_per_sample_this_epoch: float = field(
        default=-1,
        metadata={
            "help": "Average training loss per sample for the current epoch. Initial value is -1 as a placeholder."
        },
    )
    avg_valid_loss_per_sample_this_epoch: float = field(
        default=-1,
        metadata={
            "help": "Average validation loss per sample for the current epoch. Initial value is -1 as a placeholder."
        },
    )
    avg_train_loss_per_sample_this_batch: float = field(
        default=-1,
        metadata={
            "help": "Average training loss per sample for the current batch. Initial value is -1 as a placeholder."
        },
    )
    avg_valid_loss_per_sample_this_batch: float = field(
        default=-1,
        metadata={
            "help": "Average validation loss per sample for the current batch. Initial value is -1 as a placeholder."
        },
    )

    def state_dict(self) -> Dict[str, Any]:
        """Convert the State dataclass to a dictionary."""
        return asdict(self)

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load the state dictionary into the dataclass."""
        self.__dict__.update(state)
