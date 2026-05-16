"""Pytest session configuration for the OpenSCvx test suite."""

import pytest

from tests._marks import _MOREAU_OK, requires_moreau  # noqa: F401


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "moreau: tests that require a licensed moreau install",
    )
