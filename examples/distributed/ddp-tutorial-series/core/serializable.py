from __future__ import annotations

from typing import Any, Dict

__all__ = ["Serializable"]


class Serializable:
    """Interface for serialization; used by checkpointing."""

    def state_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representing the internal state.

        The returned dictionary must be pickale-able via :func:`torch.save`.

        Returns
        -------
        Dict[str, Any]: The state of the object.
        """

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restores the state of the object.

        Parameters
        ----------
        state (Dict[str, Any]): The state of the object, as previously
            returned by :meth:`.state_dict`.
        """
