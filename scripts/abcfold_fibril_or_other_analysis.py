#!/usr/bin/env python3

"""
ABCFold fibril-or-other analysis for Boltz-2, OpenFold3 and Protenix.

Analyses:

    Boltz-2:
        Aβ40 no templates
        Aβ40 with templates
        Aβ42 no templates
        Aβ42 with templates

    OpenFold3:
        Aβ40 no templates
        Aβ40 with templates
        Aβ42 no templates
        Aβ42 with templates

    Protenix:
        Aβ40 no templates
        Aβ40 with templates
        Aβ42 no templates
        Aβ42 with templates

Expected folder structure:

/home/benla/abeta_project_final/ABC_Fold/predictor_plddt_analysis/
    boltz/
        ab40/
            no_templates/
                fibril_or_other/
            with_templates/
                fibril_or_other/
        ab42/...
    openfold/...
    protenix/...

Each fibril_or_other folder should contain:

    1plddt_summary_table.csv
    2plddt_summary_table.csv
    ...
    15plddt_summary_table.csv
    20plddt_summary_table.csv
    30plddt_summary_table.csv
    50plddt_summary_table.csv

The `file` column must contain `_fibril` or `_other`.

Accepted examples:

    boltz_model_0_fibril.cif
    boltz_model_1_other.cif

    boltz_model_0.cif_fibril
    boltz_model_1.cif_other

    openfold_model_2_fibril
    protenix_model_4_other

No source files are modified.

Outputs:
    12 graphs
    12 condition tables
    1 master table
    1 condition summary table

Run:
    python abcfold_fibril_or_other_analysis.py

Validation only:
    python abcfold_fibril_or_other_analysis.py --preflight-only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(
    "/home/benla/abeta_project_final/ABC_Fold/"
    "predictor_plddt_analysis"
)

OUTPUT_ROOT = ROOT / "fibril_or_other_analysis"

MERS = list(range(1, 16)) + [20, 30, 50]

MODELS = list(range(5))

THRESHOLD = 50.0


PREDICTORS = {
    "boltz": "Boltz-2",
    "openfold": "OpenFold3",
    "protenix": "Protenix",
}


CONDITIONS = (
    (
        "ab40",
        "no_templates",
        "Aβ40",
        "without templates",
    ),
    (
        "ab40",
        "with_templates",
        "Aβ40",
        "with templates",
    ),
    (
        "ab42",
        "no_templates",
        "Aβ42",
        "without templates",
    ),
    (
        "ab42",
        "with_templates",
        "Aβ42",
        "with templates",
    ),
)


# Accept CSV or Excel versions if necessary.
TABLE_RE = re.compile(
    r"^(\d+)plddt_summary_table(?:.*)\.(csv|xlsx|xls)$",
    re.IGNORECASE,
)

MODEL_RE = re.compile(
    r"model[_\-]?([0-4])",
    re.IGNORECASE,
)


# ============================================================================
# PATH HELPERS
# ============================================================================

def dataset_folder(
    predictor: str,
    peptide: str,
    template_mode: str,
) -> Path:

    return (
        ROOT
        / predictor
        / peptide
        / template_mode
    )


def fibril_folder(
    predictor: str,
    peptide: str,
    template_mode: str,
) -> Path:

    return (
        dataset_folder(
            predictor,
            peptide,
            template_mode,
        )
        / "fibril_or_other"
    )


# ============================================================================
# INPUT READING
# ============================================================================

def read_table(path: Path) -> pd.DataFrame:

    if path.suffix.lower() == ".csv":

        return pd.read_csv(
            path,
            keep_default_na=False,
        )

    return pd.read_excel(
        path,
        keep_default_na=False,
    )


def find_mer_tables(
    folder: Path,
) -> dict[int, Path]:

    if not folder.is_dir():

        raise FileNotFoundError(
            f"Folder does not exist:\n  {folder}"
        )

    found = {}

    for path in folder.iterdir():

        if not path.is_file():
            continue

        match = TABLE_RE.match(
            path.name
        )

        if not match:
            continue

        mer = int(
            match.group(1)
        )

        if mer not in MERS:
            continue

        if mer in found:

            raise RuntimeError(
                f"Duplicate {mer}mer tables in:\n"
                f"  {folder}\n"
                f"  {found[mer]}\n"
                f"  {path}"
            )

        found[mer] = path

    missing = [
        mer
        for mer in MERS
        if mer not in found
    ]

    if missing:

        raise FileNotFoundError(
            f"Missing pLDDT tables in:\n"
            f"  {folder}\n"
            f"Missing: "
            + ", ".join(
                f"{mer}mer"
                for mer in missing
            )
        )

    return {
        mer: found[mer]
        for mer in MERS
    }


# ============================================================================
# CLASSIFICATION / MODEL PARSING
# ============================================================================

def parse_classification(
    filename: str,
) -> str | None:

    """
    Accept all sensible versions of the manually edited filename.

    Examples:

        model_0_fibril.cif
        model_0_other.cif

        model_0.cif_fibril
        model_0.cif_other

        model_0_fibril
        model_0_other
    """

    filename = filename.strip()

    patterns = (

        # Before extension:
        # model_0_fibril.cif
        r"_(fibril|other)(?=\.(?:cif|pdb|json)$)",

        # After extension:
        # model_0.cif_fibril
        r"\.(?:cif|pdb|json)_(fibril|other)$",

        # No extension:
        # model_0_fibril
        r"_(fibril|other)$",
    )

    for pattern in patterns:

        match = re.search(
            pattern,
            filename,
            re.IGNORECASE,
        )

        if match:

            return (
                match.group(1)
                .capitalize()
            )

    return None


def parse_model(
    filename: str,
) -> int | None:

    match = MODEL_RE.search(
        filename
    )

    if not match:
        return None

    return int(
        match.group(1)
    )


# ============================================================================
# READ ONE OF THE 12 DATASETS
# ============================================================================

def read_dataset(
    predictor: str,
    peptide: str,
    template_mode: str,
    predictor_label: str,
    peptide_label: str,
    template_label: str,
) -> pd.DataFrame:

    folder = fibril_folder(
        predictor,
        peptide,
        template_mode,
    )

    tables = find_mer_tables(
        folder
    )

    records = []
    errors = []

    for mer in MERS:

        path = tables[mer]

        table = read_table(
            path
        )

        required_columns = {
            "file",
            "average_atomic_plddt",
        }

        if not required_columns.issubset(
            table.columns
        ):

            raise ValueError(
                f"{path}\n"
                "must contain columns:\n"
                "  file\n"
                "  average_atomic_plddt"
            )

        if len(table) != 5:

            errors.append(
                f"{path.name}: "
                f"expected 5 rows, "
                f"found {len(table)}"
            )

        seen_models = set()

        for row_index, row in table.iterrows():

            filename = str(
                row["file"]
            ).strip()

            classification = (
                parse_classification(
                    filename
                )
            )

            model = parse_model(
                filename
            )

            if classification is None:

                errors.append(
                    f"{path.name}, "
                    f"row {row_index + 2}: "
                    "could not find "
                    "_fibril or _other in:\n"
                    f"    {filename}"
                )

                continue

            if model is None:

                errors.append(
                    f"{path.name}, "
                    f"row {row_index + 2}: "
                    "could not identify "
                    "model_0–model_4 from:\n"
                    f"    {filename}"
                )

                continue

            if model in seen_models:

                errors.append(
                    f"{path.name}: "
                    f"duplicate model_{model}"
                )

            seen_models.add(
                model
            )

            try:

                plddt = float(
                    row[
                        "average_atomic_plddt"
                    ]
                )

            except (
                TypeError,
                ValueError,
            ):

                errors.append(
                    f"{path.name}, "
                    f"row {row_index + 2}: "
                    "invalid pLDDT value"
                )

                continue

            if not 0 <= plddt <= 100:

                errors.append(
                    f"{path.name}: "
                    f"pLDDT outside 0–100: "
                    f"{plddt}"
                )

                continue

            records.append(
                {
                    "predictor": predictor,
                    "predictor_label":
                        predictor_label,

                    "peptide": peptide,
                    "peptide_label":
                        peptide_label,

                    "template_mode":
                        template_mode,

                    "template_label":
                        template_label,

                    "condition":
                        (
                            f"{peptide_label} – "
                            f"{template_label}"
                        ),

                    "nmer": mer,

                    "model": model,

                    "mean_plddt":
                        plddt,

                    "classification":
                        classification,

                    "classified_filename":
                        filename,

                    "source_table":
                        str(path),
                }
            )

        missing_models = sorted(
            set(MODELS)
            - seen_models
        )

        if missing_models:

            errors.append(
                f"{path.name}: "
                "missing "
                + ", ".join(
                    f"model_{model}"
                    for model
                    in missing_models
                )
            )

    if errors:

        shown = "\n  ".join(
            errors[:40]
        )

        if len(errors) > 40:

            shown += (
                f"\n  ...and "
                f"{len(errors) - 40} more"
            )

        raise ValueError(
            "\nErrors found in:\n"
            f"{predictor_label}, "
            f"{peptide_label}, "
            f"{template_label}\n\n"
            f"  {shown}"
        )

    data = pd.DataFrame(
        records
    ).sort_values(
        [
            "nmer",
            "model",
        ]
    )

    if len(data) != 90:

        raise RuntimeError(
            f"Expected 90 models for "
            f"{predictor_label}, "
            f"{peptide_label}, "
            f"{template_label}; "
            f"found {len(data)}"
        )

    return data


# ============================================================================
# FIGURE
# ============================================================================

def plot_dataset(
    data: pd.DataFrame,
    title: str,
    output: Path,
) -> None:

    positions = {
        mer: index
        for index, mer
        in enumerate(MERS)
    }

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    # Fixed seed reproduces the same jitter behaviour
    # as the original AF3 script.
    rng = np.random.default_rng(
        42
    )

    for (
        classification,
        marker,
        colour,
    ) in (

        (
            "Fibril",
            "^",
            "#1f77b4",
        ),

        (
            "Other",
            "o",
            "#ff7f0e",
        ),
    ):

        subset = data[
            data[
                "classification"
            ]
            == classification
        ]

        if subset.empty:
            continue

        x = np.array(
            [
                positions[nmer]
                for nmer
                in subset["nmer"]
            ],
            dtype=float,
        )

        x += rng.uniform(
            -0.18,
            0.18,
            len(x),
        )

        axis.scatter(
            x,
            subset["mean_plddt"],
            s=90,
            marker=marker,
            color=colour,
            edgecolors="black",
            linewidths=1.2,
            label=classification,
            zorder=3,
        )

    axis.axhline(
        THRESHOLD,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        label=(
            "Structured threshold (50)"
        ),
        zorder=2,
    )

    axis.set_xlim(
        -0.5,
        len(MERS) - 0.5,
    )

    axis.set_ylim(
        20,
        100,
    )

    axis.set_xticks(
        range(len(MERS))
    )

    axis.set_xticklabels(
        MERS
    )

    axis.set_yticks(
        [
            20,
            40,
            60,
            80,
            100,
        ]
    )

    axis.set_xlabel(
        "Oligomer size (N-mer)",
        fontsize=14,
    )

    axis.set_ylabel(
        "Mean atomic pLDDT",
        fontsize=14,
    )

    axis.set_title(
        title,
        fontsize=14,
    )

    axis.spines[
        "top"
    ].set_visible(False)

    axis.spines[
        "right"
    ].set_visible(False)

    legend = axis.legend(
        frameon=True,
        fancybox=False,
    )

    legend.get_frame().set_edgecolor(
        "black"
    )

    legend.get_frame().set_linewidth(
        1.5
    )

    figure.tight_layout()

    figure.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================================
# SUMMARY
# ============================================================================

def summary_table(
    master: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        predictor,
        predictor_label,
    ) in PREDICTORS.items():

        for (
            peptide,
            template_mode,
            peptide_label,
            template_label,
        ) in CONDITIONS:

            condition = master[
                (
                    master[
                        "predictor"
                    ]
                    == predictor
                )
                &
                (
                    master[
                        "peptide"
                    ]
                    == peptide
                )
                &
                (
                    master[
                        "template_mode"
                    ]
                    == template_mode
                )
            ]

            for classification in (
                "Fibril",
                "Other",
            ):

                subset = condition[
                    condition[
                        "classification"
                    ]
                    == classification
                ]

                rows.append(
                    {
                        "predictor":
                            predictor,

                        "predictor_label":
                            predictor_label,

                        "peptide":
                            peptide,

                        "peptide_label":
                            peptide_label,

                        "template_mode":
                            template_mode,

                        "template_label":
                            template_label,

                        "classification":
                            classification,

                        "model_count":
                            len(subset),

                        "above_plddt_50_count":
                            int(
                                (
                                    subset[
                                        "mean_plddt"
                                    ]
                                    >= THRESHOLD
                                ).sum()
                            ),

                        "below_plddt_50_count":
                            int(
                                (
                                    subset[
                                        "mean_plddt"
                                    ]
                                    < THRESHOLD
                                ).sum()
                            ),

                        "mean_plddt":
                            (
                                round(
                                    subset[
                                        "mean_plddt"
                                    ].mean(),
                                    2,
                                )
                                if len(subset)
                                else ""
                            ),

                        "std_plddt":
                            (
                                round(
                                    subset[
                                        "mean_plddt"
                                    ].std(),
                                    2,
                                )
                                if len(subset) > 1
                                else ""
                            ),
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# PREFLIGHT
# ============================================================================

def preflight() -> dict:

    print(
        "=" * 78
    )

    print(
        "ABCFOLD FIBRIL / OTHER PREFLIGHT"
    )

    print(
        "=" * 78
    )

    print(
        f"Root: {ROOT}"
    )

    print()

    loaded = {}

    total_tables = 0
    total_models = 0

    for predictor, predictor_label in (
        PREDICTORS.items()
    ):

        for (
            peptide,
            template_mode,
            peptide_label,
            template_label,
        ) in CONDITIONS:

            data = read_dataset(
                predictor,
                peptide,
                template_mode,
                predictor_label,
                peptide_label,
                template_label,
            )

            loaded[
                (
                    predictor,
                    peptide,
                    template_mode,
                )
            ] = data

            total_tables += 18
            total_models += len(
                data
            )

            fibrils = int(
                (
                    data[
                        "classification"
                    ]
                    == "Fibril"
                ).sum()
            )

            others = int(
                (
                    data[
                        "classification"
                    ]
                    == "Other"
                ).sum()
            )

            print(
                f"OK  "
                f"{predictor_label:<9} | "
                f"{peptide_label:<4} | "
                f"{template_label:<17} | "
                f"90/90 | "
                f"Fibril={fibrils:>2} | "
                f"Other={others:>2}"
            )

    print()

    print(
        f"Datasets validated: {len(loaded)}"
    )

    print(
        f"Input tables:       {total_tables}"
    )

    print(
        f"Models classified:  {total_models}"
    )

    print()

    print(
        "Preflight passed."
    )

    return loaded


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )

    args = parser.parse_args()

    if not ROOT.is_dir():

        raise FileNotFoundError(
            f"Root folder not found:\n"
            f"  {ROOT}"
        )

    datasets = preflight()

    if args.preflight_only:

        print()

        print(
            "Preflight-only mode complete. "
            "No output files were created."
        )

        return

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    master_tables = []

    print()

    print(
        "=" * 78
    )

    print(
        "CREATING 12 FIBRIL / OTHER FIGURES"
    )

    print(
        "=" * 78
    )

    for (
        predictor,
        predictor_label,
    ) in PREDICTORS.items():

        for (
            peptide,
            template_mode,
            peptide_label,
            template_label,
        ) in CONDITIONS:

            data = datasets[
                (
                    predictor,
                    peptide,
                    template_mode,
                )
            ].copy()

            key = (
                f"{predictor}_"
                f"{peptide}_"
                f"{template_mode}"
            )

            table_path = (
                OUTPUT_ROOT
                / (
                    f"{key}_"
                    "fibril_or_other_table.csv"
                )
            )

            graph_path = (
                OUTPUT_ROOT
                / (
                    f"{key}_"
                    "fibril_or_other_graph.png"
                )
            )

            data.to_csv(
                table_path,
                index=False,
            )

            title = (
                f"{peptide_label} – "
                f"{template_label} – "
                f"{predictor_label}: "
                "fibril classification "
                "and mean pLDDT"
            )

            plot_dataset(
                data,
                title,
                graph_path,
            )

            master_tables.append(
                data
            )

            print(
                f"Created: "
                f"{graph_path.name}"
            )

    master = pd.concat(
        master_tables,
        ignore_index=True,
    )

    master[
        "above_plddt_50"
    ] = (
        master[
            "mean_plddt"
        ]
        >= THRESHOLD
    )

    master_path = (
        OUTPUT_ROOT
        / (
            "abcfold_fibril_or_other_"
            "master_table.csv"
        )
    )

    summary_path = (
        OUTPUT_ROOT
        / (
            "abcfold_fibril_or_other_"
            "condition_summary.csv"
        )
    )

    master.to_csv(
        master_path,
        index=False,
    )

    summary_table(
        master
    ).to_csv(
        summary_path,
        index=False,
    )

    print()

    print(
        "=" * 78
    )

    print(
        "FINISHED"
    )

    print(
        "=" * 78
    )

    print(
        "Datasets analysed: 12"
    )

    print(
        "Figures created:   12"
    )

    print(
        "Models analysed:   1080"
    )

    print()

    print(
        f"Output folder:\n"
        f"  {OUTPUT_ROOT}"
    )

    print()

    print(
        f"Master table:\n"
        f"  {master_path}"
    )

    print()

    print(
        f"Summary table:\n"
        f"  {summary_path}"
    )


if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print(
            f"\nERROR: {error}",
            file=sys.stderr,
        )

        sys.exit(1)
