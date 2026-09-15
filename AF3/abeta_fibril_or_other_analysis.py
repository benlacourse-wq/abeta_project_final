"""Plot manually classified Aβ AF3 models as Fibril or Other versus mean pLDDT.

The script expects 18 per-mer CSVs for each of four conditions. It searches:
  average_plddts/fibril_or_other/   (preferred)
  average_plddts/                   (fallback)
  fibril_or_other/                  (fallback)

Each CSV needs columns `file` and `average_atomic_plddt`; each value in `file`
must end `_fibril.json` or `_other.json`, as in the alpha-synuclein workflow.

Run in Ubuntu/WSL:
  conda activate abeta_pymol
  conda install -c conda-forge pandas numpy matplotlib -y
  python abeta_fibril_or_other_analysis.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WINDOWS_AF3_ROOT = Path(
    r"C:\Users\benla\OneDrive\Documents\Bioinformatics_MSc\LIFE703\Data\AF3"
)
MERS = list(range(1, 16)) + [20, 30, 50]
CONDITIONS = (
    ("ab40", "with_templates", "Aβ40 – with templates"),
    ("ab40", "no_templates", "Aβ40 – without templates"),
    ("ab42", "with_templates", "Aβ42 – with templates"),
    ("ab42", "no_templates", "Aβ42 – without templates"),
)
THRESHOLD = 50.0
CSV_RE = re.compile(r"^(\d+)plddt_summary_table(?:.*)\.csv$", re.I)
MODEL_RE = re.compile(r"full_data_(\d+)", re.I)
CLASS_RE = re.compile(r"_(fibril|other)(?:\.json)?$", re.I)


def root_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()
    text = str(WINDOWS_AF3_ROOT)
    match = re.match(r"^([A-Za-z]):\\(.*)$", text)
    if match and sys.platform.startswith("linux"):
        drive, rest = match.groups()
        return Path("/mnt") / drive.lower() / Path(rest.replace("\\", "/"))
    return WINDOWS_AF3_ROOT


def find_csv_dir(condition_dir: Path) -> Path:
    candidates = (
        condition_dir / "average_plddts" / "fibril_or_other",
        condition_dir / "average_plddts",
        condition_dir / "fibril_or_other",
    )
    for folder in candidates:
        if folder.is_dir() and any(CSV_RE.match(p.name) for p in folder.glob("*.csv")):
            return folder
    raise FileNotFoundError(
        "No classified per-mer CSVs found. Checked:\n  "
        + "\n  ".join(map(str, candidates))
    )


def read_condition(condition_dir: Path, peptide: str, status: str, label: str) -> pd.DataFrame:
    csv_dir = find_csv_dir(condition_dir)
    by_mer = {}
    for path in csv_dir.glob("*.csv"):
        match = CSV_RE.match(path.name)
        if match and int(match.group(1)) in MERS:
            mer = int(match.group(1))
            if mer in by_mer:
                raise RuntimeError(f"Duplicate {mer}mer CSVs:\n  {by_mer[mer]}\n  {path}")
            by_mer[mer] = path

    missing = [mer for mer in MERS if mer not in by_mer]
    if missing:
        raise FileNotFoundError(
            f"{label} is missing: " + ", ".join(f"{mer}mer" for mer in missing)
        )

    records, errors = [], []
    for mer in MERS:
        path = by_mer[mer]
        table = pd.read_csv(path, keep_default_na=False)
        required = {"file", "average_atomic_plddt"}
        if not required.issubset(table.columns):
            raise ValueError(f"{path} must contain columns: file, average_atomic_plddt")
        if len(table) != 5:
            errors.append(f"{path.name}: expected 5 rows, found {len(table)}")

        seen = set()
        for row_index, row in table.iterrows():
            filename = str(row["file"]).strip()
            class_match = CLASS_RE.search(filename)
            model_match = MODEL_RE.search(filename)
            if not class_match:
                errors.append(
                    f"{path.name} row {row_index + 2}: missing _fibril/_other suffix: {filename}"
                )
                continue
            if not model_match:
                errors.append(f"{path.name} row {row_index + 2}: model number missing")
                continue
            model = int(model_match.group(1))
            if model not in range(5):
                errors.append(f"{path.name}: model_{model} is outside 0–4")
            if model in seen:
                errors.append(f"{path.name}: duplicate model_{model}")
            seen.add(model)
            try:
                plddt = float(row["average_atomic_plddt"])
            except (TypeError, ValueError):
                errors.append(f"{path.name} row {row_index + 2}: invalid pLDDT")
                continue
            records.append(
                {
                    "peptide": "Aβ40" if peptide == "ab40" else "Aβ42",
                    "template_status": status,
                    "condition": label,
                    "nmer": mer,
                    "model": model,
                    "mean_plddt": plddt,
                    "classification": class_match.group(1).capitalize(),
                    "classified_filename": filename,
                    "source_csv": str(path),
                }
            )
        if seen != set(range(5)):
            absent = sorted(set(range(5)) - seen)
            if absent:
                errors.append(f"{path.name}: missing " + ", ".join(f"model_{m}" for m in absent))

    if errors:
        shown = "\n  ".join(errors[:30])
        remainder = f"\n  ...and {len(errors)-30} more" if len(errors) > 30 else ""
        raise ValueError(f"Errors in {label} CSVs:\n  {shown}{remainder}")
    data = pd.DataFrame(records).sort_values(["nmer", "model"])
    if len(data) != 90:
        raise RuntimeError(f"Expected 90 models for {label}, found {len(data)}")
    print(f"  {label}: 18 CSVs and 90/90 classified models ({csv_dir})")
    return data


def plot_condition(data: pd.DataFrame, title: str, output: Path) -> None:
    positions = {mer: index for index, mer in enumerate(MERS)}
    figure, axis = plt.subplots(figsize=(11, 6))
    rng = np.random.default_rng(42)
    for group, marker, colour in (
        ("Fibril", "^", "#1f77b4"),
        ("Other", "o", "#ff7f0e"),
    ):
        subset = data[data["classification"] == group]
        if subset.empty:
            continue
        xs = np.array([positions[n] for n in subset["nmer"]], dtype=float)
        xs += rng.uniform(-0.18, 0.18, len(xs))
        axis.scatter(
            xs, subset["mean_plddt"], s=90, marker=marker, color=colour,
            edgecolors="black", linewidths=1.2, label=group, zorder=3,
        )
    axis.axhline(
        THRESHOLD, color="#1f77b4", linestyle="--", linewidth=1.5,
        label="Structured threshold (50)", zorder=2,
    )
    axis.set(xlim=(-0.5, len(MERS)-0.5), ylim=(20, 100))
    axis.set_xticks(range(len(MERS)), MERS)
    axis.set_yticks([20, 40, 60, 80, 100])
    axis.set_xlabel("Oligomer size (N-mer)", fontsize=14)
    axis.set_ylabel("Mean atomic pLDDT", fontsize=14)
    axis.set_title(title, fontsize=14)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    legend = axis.legend(frameon=True, fancybox=False)
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.5)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def summary_table(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, _, label in CONDITIONS:
        condition = master[master["condition"] == label]
        for group in ("Fibril", "Other"):
            subset = condition[condition["classification"] == group]
            rows.append(
                {
                    "condition": label,
                    "classification": group,
                    "model_count": len(subset),
                    "above_plddt_50_count": int((subset["mean_plddt"] >= THRESHOLD).sum()),
                    "below_plddt_50_count": int((subset["mean_plddt"] < THRESHOLD).sum()),
                    "mean_plddt": round(subset["mean_plddt"].mean(), 2) if len(subset) else "",
                    "std_plddt": round(subset["mean_plddt"].std(), 2) if len(subset) > 1 else "",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    root = root_path()
    if not root.is_dir():
        raise FileNotFoundError(f"AF3 root not found:\n  {root}")
    output = root / "fibril_or_other_analysis"
    output.mkdir(exist_ok=True)
    print(f"AF3 root: {root}\nReading manually classified pLDDT tables...")

    tables = []
    for peptide, status, label in CONDITIONS:
        data = read_condition(root / peptide / status, peptide, status, label)
        key = f"{peptide}_{status}"
        data.to_csv(output / f"{key}_fibril_or_other_table.csv", index=False)
        plot_condition(
            data, f"{label}: fibril classification and mean pLDDT",
            output / f"{key}_fibril_or_other_graph.png",
        )
        tables.append(data)

    master = pd.concat(tables, ignore_index=True)
    master["above_plddt_50"] = master["mean_plddt"] >= THRESHOLD
    master.to_csv(output / "abeta_fibril_or_other_master_table.csv", index=False)
    summary_table(master).to_csv(output / "fibril_or_other_condition_summary.csv", index=False)
    print("\nFinished: 4 plots, 4 condition tables, 1 master table and 1 summary table.")
    print(f"Output folder: {output}")


if __name__ == "__main__":
    main()
