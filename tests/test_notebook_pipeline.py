"""Meaningful synthetic checks of the EXACT self-contained notebook functions."""
import ast
import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import nbformat
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.prepare_local_data import checked_members, prepare

ROOT = Path(__file__).resolve().parents[1]


class NotebookPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = nbformat.read(ROOT / "notebooks/01_UAVid_Yerel_ve_Kaggle.ipynb", as_version=4)
        nbformat.validate(cls.notebook)
        cls.scope = {}
        # Imports/settings and definitions only; no real data or training top-level cells.
        code_cells = [c for c in cls.notebook.cells if c.cell_type == "code"]
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code_cells[0].source, cls.scope)
        for cell in code_cells[1:]:
            tree = ast.parse(cell.source)
            definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
            exec(compile(ast.Module(body=definitions, type_ignores=[]), "notebook-definitions", "exec"), cls.scope)
        torch.set_num_threads(2)

    def setUp(self):
        # Functions share globals; isolate mutable test overrides on the definition scope.
        self.s = self.scope
        self.s.update(cihaz=torch.device("cpu"), BOYUT=(32, 20), BATCH=2, EPOCH_SAYISI=1,
                      CALISMA_MODU="reset", DEVAM_CHECKPOINT="", denetim={}, VERI_MANIFESTI="synthetic")

    def test_notebook_is_output_free_and_compiles(self):
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)
                compile(cell.source, "notebook", "exec")

    def test_letterbox_alignment_padding_and_ids(self):
        image = Image.new("RGB", (8, 4), (255, 255, 255))
        label = Image.new("RGB", (8, 4), self.s["RENKLER"][7])
        x, y = self.s["donustur"](image, label)
        self.assertEqual(tuple(x.shape), (3, 20, 32))
        self.assertEqual(tuple(y.shape), (20, 32))
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.int64)
        self.assertTrue((y[:2] == 255).all())
        self.assertTrue((y[2:18] == 7).all())
        self.assertTrue((x[:, :2] == 0).all())
        self.assertTrue((x[:, 2:18] == 1).all())

    def test_unknown_source_color_cannot_disappear_in_resize(self):
        image = Image.new("RGB", (128, 80))
        label = Image.new("RGB", image.size)
        label.putpixel((1, 1), (1, 2, 3))
        with self.assertRaisesRegex(ValueError, "Tanımsız"):
            self.s["donustur"](image, label)

    def test_source_size_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            self.s["donustur"](Image.new("RGB", (8, 4)), Image.new("RGB", (8, 5)))

    def test_confusion_orientation_ignore_and_absent_class(self):
        matrix = torch.zeros(8, 8, dtype=torch.int64)
        self.s["confusion_ekle"](matrix, torch.tensor([0, 0, 7, 255]), torch.tensor([0, 7, 7, 1]))
        self.assertEqual(matrix[0, 7].item(), 1)
        self.assertEqual(matrix.sum().item(), 3)
        iou, miou = self.s["iou_hesapla"](matrix)
        self.assertEqual(iou[0].item(), 0.5)
        self.assertEqual(iou[7].item(), 0.5)
        self.assertTrue(torch.isnan(iou[1]))
        self.assertEqual(miou, 0.5)

    def test_loss_uses_valid_pixel_weights(self):
        class Fixed(nn.Module):
            def forward(self, x):
                logits = torch.zeros(x.shape[0], 8, 2, 2)
                logits[:, 0] = 2
                return logits
        x = torch.zeros(2, 3, 2, 2)
        y = torch.tensor([[[0, 0], [0, 0]], [[7, 255], [255, 255]]])
        loader = DataLoader(TensorDataset(x, y), batch_size=1)
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        result = self.s["bir_epoch"](Fixed(), loader, criterion)
        expected = criterion(Fixed()(x), y).item()
        self.assertAlmostEqual(result["loss"], expected, places=6)
        self.assertEqual(result["valid_pixels"], 5)

    def test_train_checkpoint_resume_and_unique_reset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pairs = []
            for i in range(2):
                image, label = root / f"image{i}.png", root / f"label{i}.png"
                Image.new("RGB", (8, 4), (120, 80, 40)).save(image)
                Image.new("RGB", (8, 4), self.s["RENKLER"][i]).save(label)
                pairs.append((image, label))
            self.s.update(train_pairs=pairs, val_pairs=pairs, CIKTI_KOKU=root / "runs")
            with contextlib.redirect_stdout(io.StringIO()):
                model, first, rows = self.s["egit"]()
                checkpoint = torch.load(first / "last.pt", map_location="cpu", weights_only=True)
                self.assertEqual(sum(p.numel() for p in model.parameters()), 34008)
                self.assertEqual(checkpoint["epoch"], 1)
                self.assertTrue(checkpoint["optimizer"]["state"])
                self.assertTrue((first / "history.csv").is_file())
                self.s.update(CALISMA_MODU="resume", DEVAM_CHECKPOINT=str(first / "last.pt"), EPOCH_SAYISI=2)
                _, second, resumed = self.s["egit"]()
                self.assertNotEqual(first, second)
                self.assertEqual([r["epoch"] for r in resumed], [1, 2])
                saved = torch.load(second / "best.pt", map_location="cpu", weights_only=True)
                self.s["checkpoint_uyumunu_kontrol_et"](saved)
                self.assertEqual(saved["epoch"], saved["best_epoch"])
                self.assertEqual(saved["history"][-1]["epoch"], saved["best_epoch"])
                self.assertTrue(saved["optimizer"]["state"])
                for key, tensor in saved["model"].items():
                    self.assertTrue(torch.equal(tensor, saved["best_model"][key]))
                self.s.update(CALISMA_MODU="reset", EPOCH_SAYISI=1)
                _, third, _ = self.s["egit"]()
                self.assertNotEqual(first, third)
                third_checkpoint = torch.load(third / "last.pt", map_location="cpu", weights_only=True)
                for key, tensor in checkpoint["model"].items():
                    self.assertTrue(torch.equal(tensor, third_checkpoint["model"][key]))

    def test_ambiguous_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("a", "b"):
                (root / name / "uavid_train/seq1/Images").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "tek bir"):
                self.s["split_klasoru"](root, "train")

    def test_nested_kaggle_style_split_and_matching_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seq = root / "uavid_train/uavid_train/seq1"
            (seq / "Images").mkdir(parents=True)
            (seq / "Labels").mkdir()
            Image.new("RGB", (4, 4)).save(seq / "Images/000000.png")
            Image.new("RGB", (4, 4)).save(seq / "Labels/000000.png")
            pairs, counts = self.s["ciftleri_bul"](root, "train")
            self.assertEqual(len(pairs), 1)
            self.assertEqual(counts, {"seq1": 1})
            self.assertEqual(pairs[0][0].name, pairs[0][1].name)

    def test_constant_building_reference_is_not_learned_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image, label = root / "image.png", root / "label.png"
            Image.new("RGB", (8, 4)).save(image)
            mask = Image.new("RGB", (8, 4), self.s["RENKLER"][1])
            for y in range(4):
                for x in range(4):
                    mask.putpixel((x, y), self.s["RENKLER"][7])
            mask.save(label)
            result = self.s["sabit_bina_referansi"]([(image, label)])
            self.assertEqual(result["iou"]["building"], 0.5)
            self.assertEqual(result["iou"]["human"], 0.0)
            self.assertEqual(result["miou"], 0.25)  # Only two classes have nonzero union.

    def test_checkpoint_wrong_class_order_rejected(self):
        checkpoint = {"class_map": {}, "target_size": (32, 20), "ignore_index": 255,
                      "data_manifest": "synthetic", "seed": 42, "batch_size": 2}
        with self.assertRaises(ValueError):
            self.s["checkpoint_uyumunu_kontrol_et"](checkpoint)


class ArchiveSafetyTests(unittest.TestCase):
    def test_traversal_absolute_drive_and_symlink_rejected(self):
        for name in ("../escape", "/escape", "C:/escape", "uavid_train/../../escape", "uavid_train/evil:ads"):
            with self.subTest(name=name), io.BytesIO() as buffer:
                with zipfile.ZipFile(buffer, "w") as archive:
                    archive.writestr(name, b"data")
                buffer.seek(0)
                with zipfile.ZipFile(buffer) as archive, self.assertRaises(ValueError):
                    list(checked_members(archive))
        with io.BytesIO() as buffer:
            with zipfile.ZipFile(buffer, "w") as archive:
                info = zipfile.ZipInfo("uavid_train/link")
                info.external_attr = 0o120777 << 16
                archive.writestr(info, "../../elsewhere")
            buffer.seek(0)
            with zipfile.ZipFile(buffer) as archive, self.assertRaises(ValueError):
                list(checked_members(archive))

    def test_existing_destination_is_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "existing"
            target.mkdir()
            marker = target / "keep.txt"
            marker.write_text("keep")
            with self.assertRaises(FileExistsError):
                prepare(root, target)
            self.assertEqual(marker.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
