"""Pytest configuration.

Forces the pipeline into deterministic mock mode (no API keys / network) for the whole test
session, and guarantees the project root is importable. This must run before ``pipeline.config``
is first imported, which it does — pytest imports the root ``conftest`` before collecting tests.
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PIPELINE_MOCK", "1")
# Isolate the ChromaDB memory store to a throwaway dir so the publisher's memory write-back
# (and any other store access) never touches the real ./memory/chroma_store during tests.
os.environ.setdefault("CHROMA_PERSIST_DIR", tempfile.mkdtemp(prefix="chroma_test_"))
sys.path.insert(0, str(Path(__file__).parent))
