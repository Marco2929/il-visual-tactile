# dataset_builder/__init__.py
"""Dataset Builder — Config-driven dataset generation from cached episodes."""
import sys
from pathlib import Path

_builder_dir = Path(__file__).resolve().parent
_data_dir = _builder_dir.parents[1]
for _p in (_builder_dir, _data_dir / "recording" / "record_utils", _data_dir / "record_utils"):
    if str(_p) not in sys.path and _p.exists():
        sys.path.insert(0, str(_p))