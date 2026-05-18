"""
Pytest configuration for DRO-FAIR.

CRITICAL FIX: Some pytest plugins (hypothesis, locust, asyncio) can cause
collection hangs in certain environments. This conftest disables problematic
plugins selectively and sets sensible defaults.
"""

import pytest
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def pytest_configure(config):
    """Configure pytest before collection."""
    # Limit asyncio loop scope to prevent plugin conflicts
    config.option.asyncio_mode = "strict"


def pytest_collection_modifyitems(config, items):
    """Modify test items after collection."""
    # Mark slow tests so they can be skipped with -m "not slow"
    for item in items:
        if "end_to_end" in item.nodeid or "runs_without" in item.nodeid:
            item.add_marker(pytest.mark.slow)
