"""Pytest configuration.

Forces the pipeline into deterministic mock mode (no API keys / network) for the whole test
session, and guarantees the project root is importable. This must run before ``pipeline.config``
is first imported, which it does — pytest imports the root ``conftest`` before collecting tests.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("PIPELINE_MOCK", "1")
sys.path.insert(0, str(Path(__file__).parent))
