"""Tests for Component Registry and Auto-Discovery."""

import pytest

from tdc_studio.core.exceptions import RegistryKeyError
from tdc_studio.core.registry import DATASETS, MODELS, Registry, auto_import_modules


def test_registry_registration_and_retrieval():
    test_reg = Registry("test_reg")

    @test_reg.register("mock_item")
    class MockItem:
        def __init__(self, cfg):
            self.cfg = cfg

    assert "mock_item" in test_reg.list_available()
    cls = test_reg.get("mock_item")
    assert cls is MockItem

    # Build factory
    inst = test_reg.build({"type": "mock_item", "param": 123})
    assert isinstance(inst, MockItem)
    assert inst.cfg["param"] == 123


def test_registry_duplicate_error():
    test_reg = Registry("test_dup")

    @test_reg.register("item")
    class Item1:
        pass

    with pytest.raises(KeyError):

        @test_reg.register("item")
        class Item2:
            pass


def test_registry_missing_key():
    test_reg = Registry("test_missing")
    with pytest.raises(RegistryKeyError):
        test_reg.get("non_existent")


def test_auto_import_registers_models_and_datasets():
    auto_import_modules("tdc_studio")
    assert "graph_transformer" in MODELS.list_available()
    assert "admet_loader" in DATASETS.list_available()
