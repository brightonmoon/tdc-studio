"""Core module exposing registries and base exceptions."""

from tdc_studio.core.exceptions import (
    DataPipelineError,
    ModelConfigurationError,
    RegistryKeyError,
    RemoteExecutionError,
    TDCStudioError,
)
from tdc_studio.core.registry import DATASETS, MODELS, TRANSFORMS, Registry, auto_import_modules

__all__ = [
    "Registry",
    "MODELS",
    "DATASETS",
    "TRANSFORMS",
    "auto_import_modules",
    "TDCStudioError",
    "RegistryKeyError",
    "DataPipelineError",
    "ModelConfigurationError",
    "RemoteExecutionError",
]
