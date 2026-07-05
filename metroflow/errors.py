"""Typed error hierarchy for MetroFlow.

User-facing failures (bad scenario, missing file, unknown route id, invalid CLI
input) raise a :class:`MetroFlowError` subclass with a clear, actionable message.
The CLI catches these and prints the message with a non-zero exit code, so a user
never sees a raw Python traceback for an input mistake. Genuine programming bugs
are left to propagate as ordinary exceptions.

``ConfigError`` and ``GtfsError`` also subclass :class:`ValueError`: an
out-of-range parameter or an unknown route id is a "bad value", and inheriting
from ``ValueError`` keeps the API friendly for callers that catch the standard
exception while still letting the CLI catch :class:`MetroFlowError` uniformly.
"""

from __future__ import annotations


class MetroFlowError(Exception):
    """Base class for all user-facing MetroFlow errors."""


class ConfigError(MetroFlowError, ValueError):
    """Invalid scenario configuration (bad key, out-of-range value, ...)."""


class GtfsError(MetroFlowError, ValueError):
    """A GTFS feed is missing, malformed, or lacks the requested route."""


class ScenarioFileError(MetroFlowError):
    """A scenario YAML file is missing or cannot be parsed."""


class CLIError(MetroFlowError):
    """A command-line argument combination is invalid."""


__all__ = [
    "MetroFlowError",
    "ConfigError",
    "GtfsError",
    "ScenarioFileError",
    "CLIError",
]
