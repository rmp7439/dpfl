import unittest
import os
import sys
import json
import tempfile
import shutil
import subprocess
import glob
import matplotlib.pyplot as plt
from PIL import Image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.generate_figures import (
    check_latex_toolchain,
    validate_png_dpi,
    get_pdf_dimensions_in_inches,
    save_publication_figure,
    load_dirichlet_split_validation,
    generate_dirichlet_alpha_comparison
)


class TestFigureGeneration(unittest.TestCase):
    """
    Automated test suite for Task 4: Figure generation pipeline,
    single-column width standard (3.5 in), LaTeX/PGF export, DPI validation,
    and genuine pdflatex compilation smoke tests.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_fig_")

    def tearDown(self):
        plt.close("all")
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_triple_format_export_dpi_and_width(self):
        """Verify that save_publication_figure produces non-empty .pdf (3.5 in wide), .png (>=300 DPI), and .pgf."""
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
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

        # Check DPI
        valid_dpi, actual_dpi = validate_png_dpi(paths["png"], min_dpi=300)
        self.assertTrue(valid_dpi, f"PNG DPI {actual_dpi} below 300")

        # Check PDF dimensions
        pdf_w, pdf_h = get_pdf_dimensions_in_inches(paths["pdf"])
        self.assertAlmostEqual(pdf_w, 3.5, delta=0.05, msg=f"PDF width {pdf_w} in not within 3.5 +/- 0.05 in")

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

        # Create an intentionally invalid figure that fails width or validation
        fig, ax = plt.subplots(figsize=(6.0, 4.0))  # 6.0 in violates 3.5 in requirement
        ax.plot([0, 1], [0, 1])

        with self.assertRaises(ValueError):
            save_publication_figure(fig, base_path, dpi=300, staging_dir=staging_dir)
        plt.close(fig)

        # Original file must remain untouched
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
        """Verify single-column stacked Dirichlet comparison generation with synthetic fixtures."""
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

        # Check physical PDF width is strictly 3.5 +/- 0.05 inches
        pdf_w, _ = get_pdf_dimensions_in_inches(paths["pdf"])
        self.assertAlmostEqual(pdf_w, 3.5, delta=0.05)

    def test_latex_toolchain_and_genuine_compilation(self):
        """
        Verify check_latex_toolchain finds real pdflatex and compiles
        a minimal LaTeX document importing a generated PGF figure.
        """
        has_latex, info = check_latex_toolchain()
        self.assertTrue(has_latex, f"pdflatex not found: {info}")
        self.assertIn("MiKTeX", info, f"Unexpected TeX engine info: {info}")

        # Export a real test PGF figure
        fig, ax = plt.subplots(figsize=(3.5, 2.0))
        ax.plot([0, 1], [0, 1], label=r"$\sigma=1.0$")
        ax.set_title(r"Compilation Test: $\alpha=0.1$")
        ax.legend()
        pgf_path = os.path.join(self.test_dir, "smoke.pgf")
        fig.savefig(pgf_path, format="pgf")
        plt.close(fig)

        self.assertTrue(os.path.exists(pgf_path))
        self.assertGreater(os.path.getsize(pgf_path), 0)

        # Create minimal LaTeX document that inputs the PGF
        tex_path = os.path.join(self.test_dir, "smoke_doc.tex")
        # Use forward slashes for TeX \input path
        pgf_input = pgf_path.replace("\\", "/")
        tex_content = rf"""\documentclass{{article}}
\usepackage{{pgf}}
\begin{{document}}
\input{{{pgf_input}}}
\end{{document}}
"""
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_content)

        # Compile with pdflatex
        res = subprocess.run(
            ["pdflatex", "-disable-installer", "-interaction=nonstopmode", "-halt-on-error", "smoke_doc.tex"],
            cwd=self.test_dir, capture_output=True, text=True, timeout=15
        )
        self.assertEqual(res.returncode, 0, f"pdflatex compilation failed:\n{res.stdout[-500:]}\n{res.stderr}")
        smoke_pdf = os.path.join(self.test_dir, "smoke_doc.pdf")
        self.assertTrue(os.path.exists(smoke_pdf))
        self.assertGreater(os.path.getsize(smoke_pdf), 0)

    def test_saved_pdf_widths_paper_figures(self):
        """Verify that every canonical paper figure in figures/ has width 3.5 +/- 0.05 inches."""
        fig_dir = os.path.join(project_root, "figures")
        if not os.path.exists(fig_dir):
            self.skipTest("figures/ directory does not exist yet")

        canonical_stems = [
            "fig1_stage1_distribution",
            "fig1b_dirichlet_alpha_comparison",
            "dirichlet_alpha_comparison",
            "fig2a_stage4_accuracy_convergence",
            "fig2b_stage4_loss_convergence",
            "fig3_stage4_epsilon",
            "fig4_stage5_tradeoff",
            "fig5_stage6_ablation",
            "fig5a_stage6_alpha_comparison",
            "fig5b_stage6_multiseed_summary",
            "fig6_combined"
        ]

        found = 0
        for stem in canonical_stems:
            pdf_path = os.path.join(fig_dir, f"{stem}.pdf")
            if os.path.exists(pdf_path):
                w, h = get_pdf_dimensions_in_inches(pdf_path)
                self.assertAlmostEqual(w, 3.5, delta=0.05,
                                       msg=f"{stem}.pdf width is {w:.3f} in, expected 3.50 +/- 0.05 in")
                found += 1

        if found == 0:
            self.skipTest("No canonical figures generated yet; run generate_figures.py --full first")

    def test_saved_png_dpi_paper_figures(self):
        """Verify that every canonical paper PNG in figures/ has DPI >= 300."""
        fig_dir = os.path.join(project_root, "figures")
        if not os.path.exists(fig_dir):
            self.skipTest("figures/ directory does not exist yet")

        png_files = glob.glob(os.path.join(fig_dir, "fig*.png")) + glob.glob(os.path.join(fig_dir, "dirichlet*.png"))
        if not png_files:
            self.skipTest("No canonical PNG figures found; run generate_figures.py --full first")

        for png_path in png_files:
            valid, dpi = validate_png_dpi(png_path, min_dpi=300)
            self.assertTrue(valid, f"{os.path.basename(png_path)} DPI {dpi} is below 300")

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
