"""
Reproduce the non-"fibril or other" alpha-synuclein pLDDT workflow for
Aβ40/Aβ42 AlphaFold 3 predictions, with and without templates.

Run in VSCode with:
    python abeta_af3_plddt_analysis.py

Required packages:
    pip install pandas matplotlib seaborn
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# The four folders supplied by Ben all sit below this directory.
# Change only this line if the AF3 folder is moved.
DEFAULT_AF3_ROOT = Path(
    r"C:\Users\benla\OneDrive\Documents\Bioinformatics_MSc\LIFE703\Data\AF3"
)

# An optional command-line path makes the script easy to test or reuse.
AF3_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_AF3_ROOT

CONDITIONS = (
    ("ab40", "no_templates"),
    ("ab40", "with_templates"),
    ("ab42", "no_templates"),
    ("ab42", "with_templates"),
)

MER_RE = re.compile(r"_(\d+)mer_", re.IGNORECASE)
MODEL_RE = re.compile(r"_full_data_(\d+)\.json$", re.IGNORECASE)


def mer_number(path: Path) -> int:
    """Extract the oligomer size from an AF3 prediction folder name."""
    match = MER_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not identify mer number from: {path.name}")
    return int(match.group(1))


def model_number(path: Path) -> int:
    """Extract the model number from a full_data JSON filename."""
    match = MODEL_RE.search(path.name)
    return int(match.group(1)) if match else 10**9


def find_prediction_folders(condition_dir: Path) -> list[Path]:
    """Find and numerically order all mer prediction folders."""
    folders = [
        folder
        for folder in condition_dir.iterdir()
        if folder.is_dir() and MER_RE.search(folder.name)
    ]
    return sorted(folders, key=mer_number)


def calculate_per_model_plddt(
    prediction_dir: Path, mer: int, output_dir: Path
) -> pd.DataFrame:
    """Calculate mean atomic pLDDT for every model of one oligomer size."""
    json_files = sorted(
        prediction_dir.glob("*_full_data_*.json"), key=model_number
    )
    if not json_files:
        raise FileNotFoundError(
            f"No '*_full_data_*.json' files found in {prediction_dir}"
        )

    records = []
    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        atom_plddts = data.get("atom_plddts")
        if not atom_plddts:
            print(f"  WARNING: skipped {json_file.name}: missing/empty atom_plddts")
            continue

        records.append(
            {
                "file": json_file.name,
                "average_atomic_plddt": round(
                    sum(atom_plddts) / len(atom_plddts), 2
                ),
            }
        )

    if not records:
        raise ValueError(f"No usable pLDDT data found in {prediction_dir}")

    dataframe = pd.DataFrame(records)
    dataframe.to_csv(output_dir / f"{mer}plddt_summary_table.csv", index=False)
    return dataframe


def save_individual_lineplots(
    per_mer: dict[int, pd.DataFrame], output_dir: Path, title_condition: str
) -> None:
    """Make one five-model line plot per mer (equivalent to Line_plot.py)."""
    for mer, dataframe in per_mer.items():
        labels = [f"model_{i}" for i in range(len(dataframe))]
        scores = dataframe["average_atomic_plddt"].tolist()

        fig, axis = plt.subplots()
        axis.plot(labels, scores, marker="o", color="green")
        axis.set_title(f"Model pLDDT for {title_condition} Mer {mer}")
        axis.set_xlabel("Model")
        axis.set_ylabel("Average pLDDT")
        axis.set_ylim(0, 100)
        axis.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(output_dir / f"plot_mer_{mer}_lineplot.png", dpi=300)
        plt.close(fig)


def save_combined_lineplot(
    per_mer: dict[int, pd.DataFrame], output_dir: Path, dataset_name: str
) -> None:
    """Overlay every mer's five-model profile (Combined_Line_Plot.py)."""
    fig, axis = plt.subplots(figsize=(12, 6))
    for mer, dataframe in per_mer.items():
        scores = dataframe["average_atomic_plddt"].tolist()
        labels = [f"model_{i}" for i in range(len(scores))]
        axis.plot(labels, scores, marker="o", label=f"Mer {mer}")

    axis.set_title(f"Average pLDDT Across 5 Models for Each Mer: {dataset_name}")
    axis.set_xlabel("Model")
    axis.set_ylabel("Average pLDDT")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.legend(title="Mers", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "all_mers_lineplot.png", dpi=300)
    plt.close(fig)


def save_lineplot_grid(
    per_mer: dict[int, pd.DataFrame], output_dir: Path, dataset_name: str
) -> None:
    """Make the faceted grid of per-mer model profiles (line_plot_grid.py)."""
    count = len(per_mer)
    columns = 4
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, 2.6 * rows))
    axes = axes.flatten()

    for axis, (mer, dataframe) in zip(axes, per_mer.items()):
        labels = [f"model_{i}" for i in range(len(dataframe))]
        scores = dataframe["average_atomic_plddt"].tolist()
        axis.plot(labels, scores, marker="o", color="green")
        axis.set_title(f"Mer {mer}", fontsize=10)
        axis.set_ylim(0, 100)
        axis.set_xticks(range(len(labels)), labels, rotation=45, fontsize=8)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.grid(axis="y", linestyle="--", alpha=0.5)

    for axis in axes[count:]:
        axis.set_visible(False)

    fig.suptitle(f"Average pLDDT per Model for Each Mer: {dataset_name}", fontsize=14)
    fig.text(0.04, 0.5, "Average pLDDT", va="center", rotation="vertical", fontsize=12)
    fig.tight_layout(rect=[0.05, 0.03, 1, 0.95])
    fig.savefig(output_dir / "facetwrap_all_mers_lineplots.png", dpi=300)
    plt.close(fig)


def save_boxplot(
    per_mer: dict[int, pd.DataFrame], output_dir: Path, dataset_name: str
) -> None:
    """Make pLDDT distributions by mer (Boxplots_all_model_PLDDT_per_mer.py)."""
    records = []
    for mer, dataframe in per_mer.items():
        for _, row in dataframe.iterrows():
            records.append(
                {
                    "mer_id": str(mer),
                    "model": row["file"],
                    "average_atomic_plddt": row["average_atomic_plddt"],
                }
            )

    long_data = pd.DataFrame(records)
    order = [str(mer) for mer in per_mer]
    fig, axis = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        x="mer_id",
        y="average_atomic_plddt",
        data=long_data,
        order=order,
        color="lightblue",
        fliersize=3,
        ax=axis,
    )
    axis.set_title(f"Distribution of Average pLDDT Across Models per Mer: {dataset_name}")
    axis.set_xlabel("Mer ID")
    axis.set_ylabel("Model Average pLDDT")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_model_plddt_boxplot.png", dpi=300)
    plt.close(fig)


def save_summary_and_barplot(
    per_mer: dict[int, pd.DataFrame], output_dir: Path, dataset_name: str
) -> pd.DataFrame:
    """Create merged mean/SD table and its error-bar plot."""
    rows = []
    for mer, dataframe in per_mer.items():
        values = dataframe["average_atomic_plddt"]
        rows.append(
            {
                "mer_id": mer,
                "mean_plddt": round(values.mean(), 2),
                # pandas std() matches the original sample SD calculation (ddof=1).
                "std_plddt": round(values.std(), 2),
            }
        )

    summary = pd.DataFrame(rows).sort_values("mer_id")
    summary.to_csv(output_dir / "merged_plddt_summary.csv", index=False)

    fig, axis = plt.subplots(figsize=(12, 6))
    axis.bar(
        summary["mer_id"].astype(str),
        summary["mean_plddt"],
        yerr=summary["std_plddt"],
        capsize=5,
        color="skyblue",
        edgecolor="black",
    )
    axis.set_ylabel("Mean pLDDT")
    axis.set_xlabel("Mer ID")
    axis.set_title(f"Average pLDDT per Mer (± Standard Deviation): {dataset_name}")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_mean_plddt_barplot.png", dpi=300)
    plt.close(fig)
    return summary


def analyse_condition(peptide: str, template_folder: str) -> None:
    """Run the complete α-syn-equivalent workflow for one Aβ condition."""
    condition_dir = AF3_ROOT / peptide / template_folder
    if not condition_dir.is_dir():
        raise FileNotFoundError(f"Required input folder does not exist: {condition_dir}")

    output_dir = condition_dir / "average_plddts"
    output_dir.mkdir(exist_ok=True)

    peptide_label = "Aβ40" if peptide.lower() == "ab40" else "Aβ42"
    template_label = "With Templates" if template_folder == "with_templates" else "No Templates"
    dataset_name = f"{peptide_label}, {template_label}"

    print(f"\nAnalysing {dataset_name}")
    per_mer: dict[int, pd.DataFrame] = {}
    for prediction_dir in find_prediction_folders(condition_dir):
        mer = mer_number(prediction_dir)
        dataframe = calculate_per_model_plddt(prediction_dir, mer, output_dir)
        per_mer[mer] = dataframe
        print(f"  {mer}mer: {len(dataframe)} models")

    if not per_mer:
        raise ValueError(f"No mer prediction folders found in {condition_dir}")

    per_mer = dict(sorted(per_mer.items()))
    save_individual_lineplots(per_mer, output_dir, template_label)
    save_combined_lineplot(per_mer, output_dir, dataset_name)
    save_lineplot_grid(per_mer, output_dir, dataset_name)
    save_boxplot(per_mer, output_dir, dataset_name)
    summary = save_summary_and_barplot(per_mer, output_dir, dataset_name)

    print(f"  Created {len(per_mer)} per-mer CSV tables")
    print(f"  Created merged table and {len(per_mer) + 4} figures")
    print(f"  Output: {output_dir}")
    print(summary.to_string(index=False))


def main() -> None:
    if not AF3_ROOT.is_dir():
        raise FileNotFoundError(
            f"AF3 root folder not found: {AF3_ROOT}\n"
            "Edit DEFAULT_AF3_ROOT near the top of this script if the folder moved."
        )

    print(f"AF3 root: {AF3_ROOT}")
    for peptide, template_folder in CONDITIONS:
        analyse_condition(peptide, template_folder)

    print("\nFinished all four Aβ datasets successfully.")


if __name__ == "__main__":
    main()
