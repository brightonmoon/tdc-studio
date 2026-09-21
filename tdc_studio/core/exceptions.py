"""Exceptions for TDC-Studio."""


class TDCStudioError(Exception):
    """Base exception for all TDC-Studio errors."""

    pass


class RegistryKeyError(TDCStudioError):
    """Raised when a requested component is not found in the registry."""

    pass


class DataPipelineError(TDCStudioError):
    """Raised when an error occurs during data loading or preprocessing."""

    pass


class ModelConfigurationError(TDCStudioError):
    """Raised when model architecture configuration is invalid."""

    pass


class RemoteExecutionError(TDCStudioError):
    """Raised when remote execution via Google Colab CLI fails."""

    pass
