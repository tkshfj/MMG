# protocols.py
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Protocol

__all__ = ["CalibratorProtocol", "ModelRegistryProtocol"]


# Model registry interface
class ModelRegistryProtocol(ABC):
    """
    Nominal interface for models registered in the project.
    We keep this as an ABC to preserve runtime enforcement for BaseModel subclasses.
    """
    @abstractmethod
    def build_model(self, config: Any) -> Any:
        """Return a torch.nn.Module."""
        pass

    # Output transforms (factories or callables are fine)
    @abstractmethod
    def get_cls_output_transform(self) -> Callable:
        pass

    @abstractmethod
    def get_seg_output_transform(self) -> Callable:
        pass

    @abstractmethod
    def get_auc_output_transform(self) -> Callable:
        pass

    @abstractmethod
    def get_cls_confmat_output_transform(self) -> Callable:
        pass

    @abstractmethod
    def get_seg_confmat_output_transform(self) -> Callable:
        pass

    # Capabilities
    @abstractmethod
    def get_supported_tasks(self) -> List[str]:
        """Subset of {'classification','segmentation','multitask'}."""
        pass

    # Metrics + loss + handler kwargs
    @abstractmethod
    def get_metrics(self, config: Any | None = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_loss_fn(self) -> Callable:
        pass

    @abstractmethod
    def get_handler_kwargs(self) -> Dict[str, Any]:
        """Things like num_classes, seg_output_transform, and flags for handler wiring."""
        pass


# Calibrator interface
class CalibratorProtocol(Protocol):
    """Minimal contract the evaluator expects from a calibrator."""
    t_prev: float
    cfg: Any  # should expose at least `rate_tol: float`

    def pick(
        self,
        epoch: int,
        scores: np.ndarray,
        labels: np.ndarray,
        base_rate: Optional[float],
    ) -> float:
        pass
