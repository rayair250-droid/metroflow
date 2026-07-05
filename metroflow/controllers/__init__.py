"""Controller registry / factory."""

from __future__ import annotations

from metroflow.config import ControllerConfig
from metroflow.controllers.base import Controller, InjectionCommand
from metroflow.controllers.baseline import BaselineController
from metroflow.controllers.optimizer import OptimizerController
from metroflow.controllers.predictive import PredictiveController
from metroflow.controllers.reactive import ReactiveController

_REGISTRY: dict[str, type[Controller]] = {
    BaselineController.name: BaselineController,
    ReactiveController.name: ReactiveController,
    PredictiveController.name: PredictiveController,
    OptimizerController.name: OptimizerController,
}


def available_controllers() -> list[str]:
    return list(_REGISTRY)


def make_controller(name: str, cfg: ControllerConfig) -> Controller:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown controller '{name}'. Available: {', '.join(_REGISTRY)}"
        ) from None
    return cls(cfg)


__all__ = [
    "Controller",
    "InjectionCommand",
    "BaselineController",
    "ReactiveController",
    "PredictiveController",
    "OptimizerController",
    "available_controllers",
    "make_controller",
]
