"""
Generate report-ready figures from experiment outputs.

Primary input:
- results/metrics/summary.csv

Primary output:
- results/plots/test_mse_vs_latent.png

Optional output:
- report/figures/test_mse_vs_latent.png (copied for LaTeX convenience)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for figure generation.

    Returns:
        argparse.Namespace containing parsed configuration.
    """
    parser = argparse.ArgumentParser(description="Make figures for AE vs PCA replication.")
    parser.add_argument("--summary-csv", type=str, default="results/metrics/summary.csv",
                        help="Path to consolidated summary CSV.")
    parser.add_argument("--out-dir", type=str, default="results/plots",
                        help="Directory to write generated plots.")
    parser.add_argument("--copy-to-report", action="store_true",
                        help="If set, copy figures into report/figures.")
    parser.add_argument("--report-figures-dir", type=str, default="report/figures",
                        help="Where to copy figures for LaTeX.")
    return parser.parse_args()


def _extract_curve(df: pd.DataFrame, method: str) -> Tuple[list, list]:
    """
    Extract (latent_dim, test_mse) curve for a given method from the summary dataframe.

    Args:
        df: DataFrame loaded from summary.csv.
        method: One of {"pca", "ae"}.

    Returns:
        Tuple (x, y) where:
            x: sorted list of latent dimensions
            y: corresponding list of test MSE values

    Raises:
        ValueError: If the method is not present or required columns are missing.
    """
    required = {"method", "latent_dim", "test_mse"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"summary.csv is missing required columns: {missing}")

    sub = df[df["method"] == method].copy()
    if sub.empty:
        raise ValueError(f"No rows found for method='{method}' in summary.csv")

    sub = sub.sort_values("latent_dim")
    x = sub["latent_dim"].astype(int).tolist()
    y = sub["test_mse"].astype(float).tolist()
    return x, y


def plot_test_mse_vs_latent(df: pd.DataFrame, out_path: Path) -> None:
    """
    Plot test reconstruction MSE vs latent dimension for PCA and AE.

    Args:
        df: Summary DataFrame containing methods and metrics.
        out_path: Where to save the plot (PNG).

    Returns:
        None
    """
    x_pca, y_pca = _extract_curve(df, "pca")
    x_ae, y_ae = _extract_curve(df, "ae")

    # Basic integrity: latents should match (same set)
    if x_pca != x_ae:
        print("Warning: PCA and AE latent grids differ.")
        print(f"PCA latents: {x_pca}")
        print(f"AE latents:  {x_ae}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(x_pca, y_pca, marker="o", label="PCA")
    plt.plot(x_ae, y_ae, marker="o", label="MLP Autoencoder")
    plt.xlabel("Latent dimension")
    plt.ylabel("Test reconstruction MSE")
    plt.title("MNIST reconstruction error: PCA vs Autoencoder")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def copy_to_report(src_path: Path, report_figures_dir: Path, enabled: bool) -> None:
    """
    Optionally copy a figure into the LaTeX report figures directory.

    Args:
        src_path: Source file path (generated plot).
        report_figures_dir: Destination directory for LaTeX figures.
        enabled: If True, perform the copy.

    Returns:
        None
    """
    if not enabled:
        return

    report_figures_dir.mkdir(parents=True, exist_ok=True)
    dst_path = report_figures_dir / src_path.name
    shutil.copy2(src_path, dst_path)
    print(f"Copied to report: {dst_path}")


def main() -> None:
    """
    Main entry point for figure generation.
    """
    args = parse_args()
    summary_path = Path(args.summary_csv)
    out_dir = Path(args.out_dir)

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Could not find summary CSV at {summary_path}. "
            "Run the sweep script first."
        )

    df = pd.read_csv(summary_path)

    fig_path = out_dir / "test_mse_vs_latent.png"
    plot_test_mse_vs_latent(df, fig_path)
    print(f"Wrote: {fig_path}")

    copy_to_report(fig_path, Path(args.report_figures_dir), enabled=args.copy_to_report)


if __name__ == "__main__":
    main()
