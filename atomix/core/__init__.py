"""Core module for atomix calculation and workflow classes.

Heavy scientific dependencies (ase, numpy, pymatgen, pyyaml) are imported
lazily so that ``import atomix.core`` succeeds on a minimal (click-only)
install. Accessing a specific class triggers the import of its submodule and
that submodule's scientific dependencies.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "BaseCalculation": "atomix.core.calculation",
    "Config": "atomix.core.config",
    "Workflow": "atomix.core.workflow",
    "RelaxationWorkflow": "atomix.core.workflow",
    "ScreeningWorkflowSimple": "atomix.core.workflow",
    "JobSubmitter": "atomix.core.jobs",
    "SLURMSubmitter": "atomix.core.jobs",
    "PBSSubmitter": "atomix.core.jobs",
    "LocalRunner": "atomix.core.jobs",
    "get_submitter": "atomix.core.jobs",
    "ScreeningWorkflow": "atomix.core.screening",
    "ScreeningConfig": "atomix.core.screening",
    "ScreeningResult": "atomix.core.screening",
    "AdsorptionScreening": "atomix.core.screening",
    "TrainingPoint": "atomix.core.active_learning",
    "TrainingDataExporter": "atomix.core.active_learning",
    "UncertaintyEstimator": "atomix.core.active_learning",
    "ActiveLearningSelector": "atomix.core.active_learning",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from atomix.core.active_learning import (
        ActiveLearningSelector,
        TrainingDataExporter,
        TrainingPoint,
        UncertaintyEstimator,
    )
    from atomix.core.calculation import BaseCalculation
    from atomix.core.config import Config
    from atomix.core.jobs import (
        JobSubmitter,
        LocalRunner,
        PBSSubmitter,
        SLURMSubmitter,
        get_submitter,
    )
    from atomix.core.screening import (
        AdsorptionScreening,
        ScreeningConfig,
        ScreeningResult,
        ScreeningWorkflow,
    )
    from atomix.core.workflow import (
        RelaxationWorkflow,
        ScreeningWorkflowSimple,
        Workflow,
    )
