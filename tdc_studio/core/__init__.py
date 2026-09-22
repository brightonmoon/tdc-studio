"""Core module exposing registries and base exceptions."""

from tdc_studio.core.exceptions import (
    DataPipelineError,
    EvaluationError,
    ModelConfigurationError,
    RegistryKeyError,
    RemoteExecutionError,
    ServingError,
    TDCStudioError,
)
from tdc_studio.core.registry import (
    DATASETS,
    EVALUATORS,
    MODELS,
    TRANSFORMS,
    Registry,
    auto_import_modules,
)

__all__ = [
    "Registry",
    "MODELS",
    "DATASETS",
    "TRANSFORMS",
    "EVALUATORS",
    "auto_import_modules",
    "TDCStudioError",
    "RegistryKeyError",
    "DataPipelineError",
    "ModelConfigurationError",
    "RemoteExecutionError",
    "ServingError",
    "EvaluationError",
]
