import unittest
import os
import sys
import json
import tempfile
import shutil
import matplotlib.pyplot as plt
from PIL import Image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.generate_figures import (
    check_latex_toolchain,
    validate_png_dpi,
    save_publication_figure,
    load_dirichlet_split_validation,
    generate_dirichlet_alpha_comparison
)


class TestFigureGeneration(unittest.TestCase):
    """
    Automated test suite for Task 4: Figure generation pipeline,
    LaTeX/PGF export, DPI validation, and Dirichlet alpha comparison.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_fig_")

    def tearDown(self):
        plt.close("all")
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_triple_format_export_and_dpi(self):
        """Verify that save_publication_figure produces non-empty .pdf, .png (>=300 DPI), and .pgf."""
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([1, 2], [3, 4], label=r"$\alpha=0.1$")
        ax.set_title(r"Test Plot $\sigma=1.0$")
        ax.legend()

        base_path = os.path.join(self.test_dir, "test_fig")
        paths = save_publication_figure(fig, base_path, dpi=300)
        plt.close(fig)

        self.assertTrue(os.path.exists(paths["pdf"]))
        self.assertTrue(os.path.exists(paths["png"]))
        self.assertTrue(os.path.exists(paths["pgf"]))

        self.assertGreater(os.path.getsize(paths["pdf"]), 0)
        self.assertGreater(os.path.getsize(paths["png"]), 0)
        self.assertGreater(os.path.getsize(paths["pgf"]), 0)

        valid_dpi, actual_dpi = validate_png_dpi(paths["png"], min_dpi=300)
        self.assertTrue(valid_dpi, f"PNG DPI {actual_dpi} below 300")

        # Verify PGF content contains standard LaTeX commands
        with open(paths["pgf"], "r", encoding="utf-8") as f:
            pgf_header = f.read(500)
            self.assertIn("pgf", pgf_header.lower())

    def test_staging_isolation_on_failure(self):
        """Verify that staging directory protects destination files from partial overwrites."""
        base_path = os.path.join(self.test_dir, "existing_fig")
        existing_png = f"{base_path}.png"
        with open(existing_png, "w") as f:
            f.write("original_png_content")

        staging_dir = os.path.join(self.test_dir, "staging")
        os.makedirs(staging_dir, exist_ok=True)

        # Attempt to save an invalid figure (e.g. failing DPI check by setting min_dpi unrealistically high)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        # Setting min_dpi to 1000 with dpi=300 should fail validation inside save_publication_figure
        with self.assertRaises(ValueError):
            save_publication_figure(fig, base_path, dpi=100, staging_dir=staging_dir)
        plt.close(fig)

        # Original file should be preserved
        with open(existing_png, "r") as f:
            self.assertEqual(f.read(), "original_png_content")

    def test_dirichlet_loader_valid_metadata(self):
        """Verify load_dirichlet_split_validation correctly extracts metadata and class distributions."""
        fixture_data = {
            "alpha": 0.1,
            "number_of_clients": 2,
            "class_counts_per_client": {
                "0": [10.0, 20.0, 30.0],
                "1": [30.0, 20.0, 10.0]
            }
        }
        json_path = os.path.join(self.test_dir, "valid_split.json")
        with open(json_path, "w") as f:
            json.dump(fixture_data, f)

        loaded = load_dirichlet_split_validation(json_path)
        self.assertEqual(loaded["alpha"], 0.1)
        self.assertIn("0", loaded["class_counts_per_client"])
        self.assertEqual(len(loaded["class_counts_per_client"]["0"]), 3)

    def test_dirichlet_loader_malformed_and_missing(self):
        """Verify load_dirichlet_split_validation rejects missing files or malformed keys."""
        with self.assertRaises(FileNotFoundError):
            load_dirichlet_split_validation("nonexistent_split.json")

        bad_data = {"alpha": 0.1}  # Missing class_counts_per_client
        bad_path = os.path.join(self.test_dir, "bad_split.json")
        with open(bad_path, "w") as f:
            json.dump(bad_data, f)

        with self.assertRaises(ValueError):
            load_dirichlet_split_validation(bad_path)

    def test_dirichlet_alpha_comparison_pipeline(self):
        """Verify side-by-side Dirichlet comparison generation with synthetic fixtures."""
        data_a = {
            "alpha": 0.1,
            "class_counts_per_client": {
                "0": [100.0, 0.0, 50.0],
                "1": [0.0, 150.0, 20.0]
            }
        }
        data_b = {
            "alpha": 10.0,
            "class_counts_per_client": {
                "0": [50.0, 45.0, 55.0],
                "1": [52.0, 48.0, 50.0]
            }
        }
        path_a = os.path.join(self.test_dir, "alpha01.json")
        path_b = os.path.join(self.test_dir, "alpha10.json")
        with open(path_a, "w") as f:
            json.dump(data_a, f)
        with open(path_b, "w") as f:
            json.dump(data_b, f)

        out_base = os.path.join(self.test_dir, "comparison_test")
        paths = generate_dirichlet_alpha_comparison(path_a, path_b, out_base)

        for fmt in ["pdf", "png", "pgf"]:
            self.assertTrue(os.path.exists(paths[fmt]))
            self.assertGreater(os.path.getsize(paths[fmt]), 0)

    def test_latex_toolchain_query(self):
        """Verify check_latex_toolchain returns boolean and string without raising errors."""
        has_latex, info = check_latex_toolchain()
        self.assertIsInstance(has_latex, bool)
        self.assertIsInstance(info, str)

    def test_real_split_validation_artifacts(self):
        """Verify that the repository's real alpha=0.1 and alpha=10 full split validation files exist and match."""
        a01_path = os.path.join(project_root, "results", "stage1", "split_validation_full_alpha0.1.json")
        a10_path = os.path.join(project_root, "results", "stage1", "split_validation_full_alpha10.0.json")

        self.assertTrue(os.path.exists(a01_path), "Full alpha=0.1 split artifact missing")
        self.assertTrue(os.path.exists(a10_path), "Full alpha=10.0 split artifact missing")

        d01 = load_dirichlet_split_validation(a01_path)
        d10 = load_dirichlet_split_validation(a10_path)

        self.assertEqual(d01.get("alpha"), 0.1)
        self.assertEqual(d10.get("alpha"), 10.0)
        self.assertEqual(len(d01["class_counts_per_client"]), 5)
        self.assertEqual(len(d10["class_counts_per_client"]), 5)


if __name__ == "__main__":
    unittest.main()
