#!/usr/bin/env python3
"""
ABCFold predictor-specific pLDDT analysis for Aβ40/Aβ42.

Reproduces the plotting workflow used by abeta_af3_plddt_analysis.py, but
separately for:

    Boltz     × Aβ40/Aβ42 × no_templates/with_templates
    OpenFold  × Aβ40/Aβ42 × no_templates/with_templates
    Protenix  × Aβ40/Aβ42 × no_templates/with_templates

= 12 predictor/condition datasets.

The script reads the canonical ABCFold output CIFs:

    <condition>/output_models/boltz_model_0.cif ... boltz_model_4.cif
    <condition>/output_models/openfold_model_0.cif ... openfold_model_4.cif
    <condition>/output_models/protenix_model_0.cif ... protenix_model_4.cif

pLDDT is read from the mmCIF _atom_site.B_iso_or_equiv column. This provides
one consistent method for all three predictors and does not depend on optional
ReactIFpTM/full-data JSON files.

Expected oligomer sizes:
    1-15mer, 20mer, 30mer, 50mer

For each of the 12 datasets the script creates:
    - 18 per-mer CSV tables
    - 18 per-mer five-model line plots
    - all_mers_lineplot.png
    - facetwrap_all_mers_lineplots.png
    - plot_model_plddt_boxplot.png
    - plot_mean_plddt_barplot.png
    - merged_plddt_summary.csv

It also creates:
    - MASTER_plddt_summary.csv
    - MASTER_model_plddt_values.csv

Default input:
    /home/benla/abeta_project_final/ABC_Fold/ab40_ab42_results

Default output:
    /home/benla/abeta_project_final/ABC_Fold/predictor_plddt_analysis

Run:
    python abcfold_predictor_plddt_analysis.py

Optional:
    python abcfold_predictor_plddt_analysis.py --preflight-only
    python abcfold_predictor_plddt_analysis.py --predictor boltz
    python abcfold_predictor_plddt_analysis.py --predictor openfold
    python abcfold_predictor_plddt_analysis.py --predictor protenix
"""

from __future__ import annotations

import argparse
import math
import shlex
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_RESULTS_ROOT = Path(
    "/home/benla/abeta_project_final/ABC_Fold/ab40_ab42_results"
)

MERS = list(range(1, 16)) + [20, 30, 50]
MODELS = list(range(5))

PREDICTORS = {
    "boltz": "Boltz-2",
    "openfold": "OpenFold3",
    "protenix": "Protenix",
}

CONDITIONS = (
    ("ab40", "no_templates", "Aβ40", "No Templates"),
    ("ab40", "with_templates", "Aβ40", "With Templates"),
    ("ab42", "no_templates", "Aβ42", "No Templates"),
    ("ab42", "with_templates", "Aβ42", "With Templates"),
)


# -----------------------------------------------------------------------------
# ABCFold paths
# -----------------------------------------------------------------------------

def condition_directory(
    results_root: Path,
    peptide: str,
    template_mode: str,
    mer: int,
) -> Path:
    """Return the canonical ABCFold directory for one peptide/mer/condition."""

    peptide_upper = peptide.upper()

    if template_mode == "no_templates":
        group = f"ABCFold_{peptide_upper}_no_templates"
        condition = f"{peptide}_{mer}mer_no_templates"
    elif template_mode == "with_templates":
        group = f"ABCFold_{peptide_upper}_templates"
        condition = f"{peptide}_{mer}mer_with_templates"
    else:
        raise ValueError(f"Unknown template mode: {template_mode}")

    return results_root / group / condition


def output_cif(
    results_root: Path,
    peptide: str,
    template_mode: str,
    mer: int,
    predictor: str,
    model: int,
) -> Path:
    """Return the canonical output_models CIF for one model."""

    return (
        condition_directory(results_root, peptide, template_mode, mer)
        / "output_models"
        / f"{predictor}_model_{model}.cif"
    )


# -----------------------------------------------------------------------------
# mmCIF pLDDT extraction
# -----------------------------------------------------------------------------

def mean_atomic_plddt_from_cif(cif_path: Path) -> float:
    """
    Calculate mean atomic pLDDT from _atom_site.B_iso_or_equiv in an mmCIF.

    Only ATOM records are included, which is appropriate for these protein-only
    Aβ predictions and prevents unrelated HETATM records from affecting the mean.

    No third-party structural parser is required.
    """

    try:
        lines = cif_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        raise RuntimeError(f"Could not read CIF: {cif_path}") from error

    i = 0

    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue

        j = i + 1
        headers: list[str] = []

        while j < len(lines) and lines[j].lstrip().startswith("_"):
            # mmCIF header lines can contain comments/spacing; the field name is
            # always the first token.
            headers.append(lines[j].strip().split()[0])
            j += 1

        b_field = "_atom_site.B_iso_or_equiv"
        if b_field not in headers:
            i = max(i + 1, j)
            continue

        b_index = headers.index(b_field)
        group_index = (
            headers.index("_atom_site.group_PDB")
            if "_atom_site.group_PDB" in headers
            else None
        )

        n_columns = len(headers)
        token_buffer: list[str] = []
        values: list[float] = []

        while j < len(lines):
            stripped = lines[j].strip()

            # End of this loop block.
            if stripped == "loop_" or stripped.startswith("_") or stripped.startswith("#"):
                break

            if not stripped:
                j += 1
                continue

            try:
                tokens = shlex.split(lines[j], comments=False, posix=True)
            except ValueError as error:
                raise RuntimeError(
                    f"Could not tokenize atom-site row in {cif_path}:\n{lines[j]}"
                ) from error

            token_buffer.extend(tokens)

            while len(token_buffer) >= n_columns:
                row = token_buffer[:n_columns]
                token_buffer = token_buffer[n_columns:]

                if group_index is not None and row[group_index].upper() != "ATOM":
                    continue

                raw_value = row[b_index]
                if raw_value in {"?", "."}:
                    continue

                try:
                    value = float(raw_value)
                except ValueError:
                    continue

                values.append(value)

            j += 1

        if values:
            mean_value = sum(values) / len(values)

            if not (0 <= mean_value <= 100):
                raise ValueError(
                    f"Mean B_iso_or_equiv outside pLDDT range 0-100 for {cif_path}: "
                    f"{mean_value:.3f}"
                )

            return mean_value

        raise ValueError(
            f"Found _atom_site.B_iso_or_equiv but no usable ATOM values in: {cif_path}"
        )

    raise ValueError(
        f"No _atom_site.B_iso_or_equiv atom-site column found in: {cif_path}"
    )


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def preflight(results_root: Path, predictors: list[str]) -> None:
    """Require all expected output-model CIFs before starting the analysis."""

    print("=" * 78)
    print("ABCFOLD pLDDT PREFLIGHT")
    print("=" * 78)
    print(f"Results root: {results_root}")
    print()

    missing: list[Path] = []
    total = 0

    for predictor in predictors:
        for peptide, template_mode, _, _ in CONDITIONS:
            for mer in MERS:
                for model in MODELS:
                    total += 1
                    path = output_cif(
                        results_root,
                        peptide,
                        template_mode,
                        mer,
                        predictor,
                        model,
                    )
                    if not path.is_file():
                        missing.append(path)

    print(f"Expected output CIFs: {total}")
    print(f"Found:                {total - len(missing)}")
    print(f"Missing:              {len(missing)}")

    if missing:
        print("\nMissing files:")
        for path in missing:
            print(f"  {path}")
        raise FileNotFoundError(
            f"Preflight failed: {len(missing)} expected output-model CIF(s) are missing."
        )

    print("Preflight passed: every requested output-model CIF is present.\n")


# -----------------------------------------------------------------------------
# Tables and plots
# -----------------------------------------------------------------------------

def calculate_per_model_plddt(
    results_root: Path,
    peptide: str,
    template_mode: str,
    predictor: str,
    mer: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Calculate mean atomic pLDDT for models 0-4 of one mer."""

    rows = []

    for model in MODELS:
        cif = output_cif(
            results_root,
            peptide,
            template_mode,
            mer,
            predictor,
            model,
        )

        score = mean_atomic_plddt_from_cif(cif)

        rows.append(
            {
                "model": f"model_{model}",
                "model_index": model,
                "file": cif.name,
                "source_path": str(cif),
                "average_atomic_plddt": round(score, 2),
            }
        )

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_dir / f"{mer}plddt_summary_table.csv", index=False)
    return dataframe


def save_individual_lineplots(
    per_mer: dict[int, pd.DataFrame],
    output_dir: Path,
    dataset_name: str,
) -> None:
    """One five-model line plot for every mer."""

    for mer, dataframe in per_mer.items():
        labels = dataframe["model"].tolist()
        scores = dataframe["average_atomic_plddt"].tolist()

        fig, axis = plt.subplots(figsize=(7, 5))
        axis.plot(labels, scores, marker="o")
        axis.set_title(f"Model pLDDT for {dataset_name}: Mer {mer}")
        axis.set_xlabel("Model")
        axis.set_ylabel("Average pLDDT")
        axis.set_ylim(0, 100)
        axis.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(output_dir / f"plot_mer_{mer}_lineplot.png", dpi=300)
        plt.close(fig)


def save_combined_lineplot(
    per_mer: dict[int, pd.DataFrame],
    output_dir: Path,
    dataset_name: str,
) -> None:
    """Overlay all 18 mer profiles, matching the supplied AF3 figure."""

    fig, axis = plt.subplots(figsize=(12, 6))

    for mer, dataframe in per_mer.items():
        axis.plot(
            dataframe["model"],
            dataframe["average_atomic_plddt"],
            marker="o",
            label=f"Mer {mer}",
        )

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
    per_mer: dict[int, pd.DataFrame],
    output_dir: Path,
    dataset_name: str,
) -> None:
    """Faceted grid showing the five-model profile for each mer separately."""

    count = len(per_mer)
    columns = 4
    rows = math.ceil(count / columns)

    fig, axes = plt.subplots(rows, columns, figsize=(16, 2.6 * rows))
    axes = axes.flatten()

    for axis, (mer, dataframe) in zip(axes, per_mer.items()):
        labels = dataframe["model"].tolist()
        scores = dataframe["average_atomic_plddt"].tolist()

        axis.plot(range(len(labels)), scores, marker="o")
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
    per_mer: dict[int, pd.DataFrame],
    output_dir: Path,
    dataset_name: str,
) -> None:
    """Distribution of the five model-average pLDDT values for every mer."""

    labels = [str(mer) for mer in per_mer]
    values = [
        dataframe["average_atomic_plddt"].tolist()
        for dataframe in per_mer.values()
    ]

    fig, axis = plt.subplots(figsize=(12, 6))
    axis.boxplot(values, showfliers=True)
    axis.set_xticks(range(1, len(labels) + 1))
    axis.set_xticklabels(labels)
    axis.set_title(f"Distribution of Average pLDDT Across Models per Mer: {dataset_name}")
    axis.set_xlabel("Mer ID")
    axis.set_ylabel("Model Average pLDDT")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_model_plddt_boxplot.png", dpi=300)
    plt.close(fig)


def save_summary_and_barplot(
    per_mer: dict[int, pd.DataFrame],
    output_dir: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """Create mean ± sample SD summary and bar plot."""

    rows = []

    for mer, dataframe in per_mer.items():
        values = dataframe["average_atomic_plddt"]

        rows.append(
            {
                "mer_id": mer,
                "mean_plddt": round(values.mean(), 2),
                "std_plddt": round(values.std(ddof=1), 2),
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


# -----------------------------------------------------------------------------
# One predictor/condition dataset
# -----------------------------------------------------------------------------

def analyse_dataset(
    results_root: Path,
    output_root: Path,
    predictor: str,
    peptide: str,
    template_mode: str,
    peptide_label: str,
    template_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full plotting workflow for one predictor/peptide/template dataset."""

    predictor_label = PREDICTORS[predictor]
    dataset_name = f"{peptide_label}, {template_label} — {predictor_label}"

    output_dir = output_root / predictor / peptide / template_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAnalysing {dataset_name}")
    print(f"  Output: {output_dir}")

    per_mer: dict[int, pd.DataFrame] = {}
    master_model_rows = []

    for mer in MERS:
        dataframe = calculate_per_model_plddt(
            results_root,
            peptide,
            template_mode,
            predictor,
            mer,
            output_dir,
        )
        per_mer[mer] = dataframe

        print(
            f"  {mer:>2}mer: "
            + ", ".join(
                f"model_{int(row.model_index)}={row.average_atomic_plddt:.2f}"
                for row in dataframe.itertuples()
            )
        )

        for row in dataframe.to_dict(orient="records"):
            master_model_rows.append(
                {
                    "predictor": predictor,
                    "predictor_label": predictor_label,
                    "peptide": peptide,
                    "peptide_label": peptide_label,
                    "template_mode": template_mode,
                    "template_label": template_label,
                    "mer_id": mer,
                    **row,
                }
            )

    save_individual_lineplots(per_mer, output_dir, dataset_name)
    save_combined_lineplot(per_mer, output_dir, dataset_name)
    save_lineplot_grid(per_mer, output_dir, dataset_name)
    save_boxplot(per_mer, output_dir, dataset_name)
    summary = save_summary_and_barplot(per_mer, output_dir, dataset_name)

    summary.insert(0, "template_label", template_label)
    summary.insert(0, "template_mode", template_mode)
    summary.insert(0, "peptide_label", peptide_label)
    summary.insert(0, "peptide", peptide)
    summary.insert(0, "predictor_label", predictor_label)
    summary.insert(0, "predictor", predictor)

    print(f"  Created 18 per-mer CSV tables")
    print(f"  Created 22 figures (18 individual + 4 summary figures)")
    print(f"  Created merged_plddt_summary.csv")

    return summary, pd.DataFrame(master_model_rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create predictor-specific ABCFold Aβ40/Aβ42 pLDDT figures."
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"ABCFold ab40_ab42_results directory (default: {DEFAULT_RESULTS_ROOT})",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: <ABC_Fold>/predictor_plddt_analysis"
        ),
    )

    parser.add_argument(
        "--predictor",
        choices=["all", *PREDICTORS.keys()],
        default="all",
        help="Analyse all predictors or only one predictor (default: all).",
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check all expected CIFs and exit without creating figures.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_root = args.results_root.expanduser().resolve()

    if not results_root.is_dir():
        raise FileNotFoundError(f"ABCFold results root not found: {results_root}")

    if args.output_root is None:
        output_root = results_root.parent / "predictor_plddt_analysis"
    else:
        output_root = args.output_root.expanduser().resolve()

    predictors = (
        list(PREDICTORS)
        if args.predictor == "all"
        else [args.predictor]
    )

    preflight(results_root, predictors)

    if args.preflight_only:
        print("Preflight-only mode complete. No figures or CSV files were created.")
        return

    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("ABCFOLD PREDICTOR-SPECIFIC pLDDT ANALYSIS")
    print("=" * 78)
    print(f"Input:  {results_root}")
    print(f"Output: {output_root}")
    print(f"Predictors: {', '.join(PREDICTORS[p] for p in predictors)}")

    master_summaries = []
    master_models = []
    dataset_count = 0

    for predictor in predictors:
        for peptide, template_mode, peptide_label, template_label in CONDITIONS:
            dataset_count += 1
            summary, model_values = analyse_dataset(
                results_root,
                output_root,
                predictor,
                peptide,
                template_mode,
                peptide_label,
                template_label,
            )
            master_summaries.append(summary)
            master_models.append(model_values)

    master_summary = pd.concat(master_summaries, ignore_index=True)
    master_model_values = pd.concat(master_models, ignore_index=True)

    master_summary.to_csv(
        output_root / "MASTER_plddt_summary.csv",
        index=False,
    )
    master_model_values.to_csv(
        output_root / "MASTER_model_plddt_values.csv",
        index=False,
    )

    print("\n" + "=" * 78)
    print("FINISHED")
    print("=" * 78)
    print(f"Datasets analysed: {dataset_count}")
    print(f"Mers per dataset:  {len(MERS)}")
    print(f"Models per mer:     {len(MODELS)}")
    print(f"Model CIFs analysed:{dataset_count * len(MERS) * len(MODELS):>5}")
    print(f"Output root:        {output_root}")
    print()
    print("Master tables:")
    print(f"  {output_root / 'MASTER_plddt_summary.csv'}")
    print(f"  {output_root / 'MASTER_model_plddt_values.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
