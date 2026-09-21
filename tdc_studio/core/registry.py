"""Component Registry for loose coupling across datasets, models, and transforms."""

import importlib
import pkgutil
from typing import Any, Callable, Dict, List, Optional

from tdc_studio.core.exceptions import RegistryKeyError


class Registry:
    """Dynamic component registry supporting auto-discovery and factory instantiation."""

    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Callable[..., Any]] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, name: Optional[str] = None):
        """Decorator to register a class or function."""

        def _register(cls: Callable[..., Any]):
            key = name if name is not None else cls.__name__
            if key in self._module_dict:
                raise KeyError(f"'{key}' is already registered in registry '{self._name}'.")
            self._module_dict[key] = cls
            return cls

        return _register

    def get(self, key: str) -> Callable[..., Any]:
        """Retrieve registered class or constructor."""
        if key not in self._module_dict:
            available = list(self._module_dict.keys())
            raise RegistryKeyError(
                f"'{key}' not found in registry '{self._name}'. Available: {available}"
            )
        return self._module_dict[key]

    def build(self, cfg: Dict[str, Any], **kwargs) -> Any:
        """Instantiate an object using a configuration dictionary with a 'type' field."""
        cfg_copy = cfg.copy()
        if "type" not in cfg_copy:
            raise KeyError(f"Configuration must specify 'type' field to build from {self._name}.")
        obj_type = cfg_copy.pop("type")
        obj_cls = self.get(obj_type)
        return obj_cls({**cfg_copy, **kwargs})

    def list_available(self) -> List[str]:
        """Return list of all registered keys."""
        return sorted(list(self._module_dict.keys()))


# Pre-defined registries
MODELS = Registry("models")
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")


def auto_import_modules(package_name: str) -> None:
    """Walk packages and import all modules to trigger @register decorators."""
    try:
        package = importlib.import_module(package_name)
    except ModuleNotFoundError:
        return

    if not hasattr(package, "__path__"):
        return

    for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            importlib.import_module(module_name)
        except Exception:
            # Continue scanning even if optional dependencies fail
            pass
