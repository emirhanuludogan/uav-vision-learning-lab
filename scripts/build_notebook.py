"""Build a standalone notebook from readable # %% source; no runtime local imports."""
import hashlib
import json
from pathlib import Path


def build():
    root = Path(__file__).resolve().parents[1]
    source = root / "notebooks" / "uavid_notebook.py"
    cells = []
    kind, lines = None, []

    def flush():
        if kind is None:
            return
        text = "".join(lines).strip() + "\n"
        if kind == "markdown":
            text = "\n".join(line[2:] if line.startswith("# ") else
                             ("" if line == "#" else line) for line in text.splitlines()) + "\n"
        cell = {"cell_type": kind, "metadata": {},
                "id": hashlib.sha256((str(len(cells)) + text).encode()).hexdigest()[:12],
                "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    for line in source.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            lines = []
        else:
            lines.append(line)
    flush()
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"}},
        "nbformat": 4, "nbformat_minor": 5}
    output = source.with_name("01_UAVid_Yerel_ve_Kaggle.ipynb")
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    build()
