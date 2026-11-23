"""Every text file the course reads or writes, and every text a subprocess returns, is
decoded as UTF-8 by name. Without `encoding=` Python falls back to the system code page,
cp1252 on Windows, which has no `├─` for a torchinfo summary nor `é` for a comment."""

import ast
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOT_FILES = {"webbrowser", "Image", "tarfile", "zipfile", "gzip", "h5py", "os", "io"}
SUBPROCESS = {"run", "Popen", "check_output", "check_call", "call"}


def text_mode(call: ast.Call) -> bool:
    """Whether `call` opens a file or a pipe as text and leaves the encoding to the OS."""
    if any(k.arg == "encoding" for k in call.keywords):
        return False
    f = call.func
    if isinstance(f, ast.Attribute) and f.attr in {"read_text", "write_text"}:
        return True
    builtin = isinstance(f, ast.Name) and f.id == "open"
    method = (
        isinstance(f, ast.Attribute)
        and f.attr == "open"
        and not (isinstance(f.value, ast.Name) and f.value.id in NOT_FILES)
    )
    if builtin or method:
        mode = next((k.value for k in call.keywords if k.arg == "mode"), None)
        position = 1 if builtin else 0
        if mode is None and len(call.args) > position:
            mode = call.args[position]
        if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
            return "b" not in mode.value
        return mode is None
    if isinstance(f, ast.Attribute) and f.attr in SUBPROCESS:
        return any(
            k.arg in {"text", "universal_newlines"}
            and isinstance(k.value, ast.Constant)
            and k.value.value is True
            for k in call.keywords
        )
    return False


def sources() -> list[tuple[str, str]]:
    """(place, code) for every tracked Python file and every notebook code cell."""
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "*.py", "*.ipynb"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout.split()
    out = []
    for rel in tracked:
        text = (ROOT / rel).read_text(encoding="utf-8")
        if rel.endswith(".py"):
            out.append((rel, text))
            continue
        for i, cell in enumerate(json.loads(text)["cells"]):
            if cell["cell_type"] == "code":
                out.append((f"{rel} cell {i}", "".join(cell["source"])))
    return out


def test_text_io_names_its_encoding():
    missing = []
    for place, code in sources():
        try:
            tree = ast.parse(code)
        except SyntaxError:  # a notebook cell with a magic
            continue
        missing += [
            f"{place}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and text_mode(node)
        ]
    assert not missing, 'add encoding="utf-8" at:\n' + "\n".join(missing)
