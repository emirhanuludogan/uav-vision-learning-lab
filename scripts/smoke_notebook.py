"""Execute the standalone notebook in a clean CPU kernel; never start training."""
import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-audit", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ["UAVID_DATA_ROOT"] = str(root / "data" / "official")
    notebook = nbformat.read(root / "notebooks" / "01_UAVid_Yerel_ve_Kaggle.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "notebook", "exec")
    index = next(i for i, cell in enumerate(notebook.cells)
                 if cell.cell_type == "code" and "CIHAZ_SECIMI =" in cell.source)
    notebook.cells.insert(index + 1, nbformat.v4.new_code_cell(
        "cihaz = torch.device('cpu')\ntorch.set_num_threads(2)\n"
        "EGITIMI_BASLAT = False\n"
        f"TAM_DENETIM = {args.full_audit!r}\n"
        "print('SMOKE: CPU, eğitim kapalı; gerçek yerel veri.')"))
    print("Clean kernel execution starting; training disabled.", flush=True)
    NotebookClient(notebook, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(root)}}).execute()
    output = root / "reports" / "notebook-smoke.executed.ipynb"
    output.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, output)
    for cell in notebook.cells:
        for item in cell.get("outputs", []):
            if item.output_type == "stream":
                print(item.text)
    print("Executed notebook saved locally:", output)


if __name__ == "__main__":
    main()
