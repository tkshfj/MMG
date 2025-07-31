# model_protocol.py
from typing import Any, Callable, Dict, List
from abc import ABC, abstractmethod


class ModelContextProtocol(ABC):
    @abstractmethod
    def build_model(self, config: Any) -> Any:
        pass

    @abstractmethod
    def get_output_transform(self) -> Callable:
        pass

    @abstractmethod
    def get_supported_tasks(self) -> List[str]:
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_loss_fn(self) -> Callable:
        pass

    @abstractmethod
    def get_handler_kwargs(self) -> Dict[str, Any]:
        pass
