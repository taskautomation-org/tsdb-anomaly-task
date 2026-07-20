"""Run every docstring example in the package."""

from __future__ import annotations

import doctest
import importlib
import pkgutil

import pytest

import tsdb_anomaly_task

MODULES = sorted(
    module.name
    for module in pkgutil.walk_packages(tsdb_anomaly_task.__path__, prefix="tsdb_anomaly_task.")
    if not module.name.endswith("__main__")
)


@pytest.mark.parametrize("name", ["tsdb_anomaly_task", *MODULES])
def test_docstring_examples(name: str) -> None:
    module = importlib.import_module(name)
    results = doctest.testmod(module, verbose=False, raise_on_error=False)
    assert results.failed == 0, f"{results.failed} doctest failure(s) in {name}"
