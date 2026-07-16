from __future__ import annotations

import importlib.util
import io
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_environment.py"


def _load_check_environment():
    spec = importlib.util.spec_from_file_location("check_environment", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_safe_print_replaces_text_unencodable_by_console() -> None:
    safe_print = _load_check_environment().safe_print
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict")

    safe_print("Filesystem C:\\tmp: not writable: refused: 拒绝访问", stream=stream)
    stream.flush()

    output = buffer.getvalue().decode("cp1252")
    assert output.startswith("Filesystem C:\\tmp: not writable: refused: ")
    assert "?" in output
