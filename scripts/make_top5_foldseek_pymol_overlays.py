#!/usr/bin/env python3
"""
Create ten publication-style PyMOL overlay panel figures from the FINAL,
peptide-separated Aβ40/Aβ42 Foldseek analyses.

The selection logic is deliberately identical to the final Foldseek assignment logic:
  1) highest query-normalised TM score (qTM)
  2) highest query coverage (qcov)
  3) longest alignment (alnlen)
  4) lowest Foldseek RMSD

Outputs
-------
Figure 01  top 5 AlphaFold 3 Aβ40 predictions vs experimental Aβ40
Figure 02  top 5 AlphaFold 3 Aβ42 predictions vs experimental Aβ42
Figure 03  top 5 Boltz Aβ40 predictions vs experimental Aβ40
Figure 04  top 5 Boltz Aβ42 predictions vs experimental Aβ42
Figure 05  top 5 Protenix Aβ40 predictions vs experimental Aβ40
Figure 06  top 5 Protenix Aβ42 predictions vs experimental Aβ42
Figure 07  top 5 OpenFold Aβ40 predictions vs experimental Aβ40
Figure 08  top 5 OpenFold Aβ42 predictions vs experimental Aβ42
Figure 09  top 5 predictions across all four methods for Aβ40
Figure 10  top 5 predictions across all four methods for Aβ42

Each panel shows the SAME representative prediction chain that was used by Foldseek,
superposed on its assigned experimental representative chain. Foldseek qTM and alnTM
are read directly from the final assignment table. Contact Jaccard is calculated from
Cα contact sets on the exact two representative-chain PDBs (8 Å cutoff; residue pairs
separated by >2 sequence positions).

Source structures are READ ONLY. The script writes only under:
  <project>/analysis/foldseek/top5_structural_overlays
  <project>/figures/foldseek/top5_structural_overlays

Recommended environment:
  conda activate abeta_pymol

Usage:
  python make_top5_foldseek_pymol_overlays.py --preflight
  python make_top5_foldseek_pymol_overlays.py --all

Optional:
  python make_top5_foldseek_pymol_overlays.py --all --force
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_PROJECT = Path("/home/benla/abeta_project_final")
METHODS = ["AlphaFold", "Boltz", "Protenix", "OpenFold"]
METHOD_DISPLAY = {
    "AlphaFold": "AlphaFold 3",
    "Boltz": "Boltz-2",
    "Protenix": "Protenix-v2",
    "OpenFold": "OpenFold3",
}
PEPTIDE_DISPLAY = {"ab40": "Aβ40", "ab42": "Aβ42"}

# Same selection precedence as the final Foldseek pipeline.
RANK_COLS = ["qtmscore", "qcov", "alnlen", "rmsd"]
RANK_ASC = [False, False, False, True]

# Contact-map definition for the requested displayed Contact Jaccard.
CONTACT_CUTOFF_A = 8.0
MIN_SEQUENCE_SEPARATION = 3  # |i-j| >= 3; excludes i/i+1 and i/i+2 backbone-neighbour contacts

# Rendering / figure style
PRED_COLOR = "cyan"
EXP_COLOR = "salmon"
PANEL_WIDTH = 1050
PANEL_HEIGHT = 720
RAY_TRACE = True

FIG_LAYOUT = [
    (0, slice(0, 2)),
    (0, slice(2, 4)),
    (0, slice(4, 6)),
    (1, slice(1, 3)),
    (1, slice(3, 5)),
]
LETTERS = ["A", "B", "C", "D", "E"]


# -----------------------------------------------------------------------------
# Paths / normalisation
# -----------------------------------------------------------------------------

def analysis_dir(project: Path, peptide: str) -> Path:
    return project / "analysis" / "foldseek" / f"{peptide}_separate"


def assignment_path(project: Path, peptide: str) -> Path:
    return analysis_dir(project, peptide) / "assignments" / "model_level_assignments_720.tsv"


def manifest_path(project: Path, peptide: str) -> Path:
    n = 753 if peptide == "ab40" else 766
    return analysis_dir(project, peptide) / f"all_structure_manifest_{n}.tsv"


def output_analysis_dir(project: Path) -> Path:
    return project / "analysis" / "foldseek" / "top5_structural_overlays"


def output_figure_dir(project: Path) -> Path:
    return project / "figures" / "foldseek" / "top5_structural_overlays"


def normalize_id(value: str) -> str:
    return str(value).strip().lower().replace(".pdb", "")


def clean_pdb_label(target: str) -> str:
    # Assignment table already provides best_pdb_id, but keep a robust fallback.
    s = str(target)
    m = re.match(r"(?i)^([0-9][A-Za-z0-9]{3})", s)
    return m.group(1).upper() if m else s


def template_display(value: str) -> str:
    return "templates" if value == "with_templates" else "no templates"


# -----------------------------------------------------------------------------
# Data loading / exact selection
# -----------------------------------------------------------------------------

def load_assignments(project: Path, peptide: str) -> pd.DataFrame:
    path = assignment_path(project, peptide)
    if not path.is_file():
        raise FileNotFoundError(f"Assignment table not found: {path}")
    df = pd.read_csv(path, sep="\t")

    required = {
        "foldseek_id", "method", "peptide", "template_status", "oligomer_size",
        "model_rank", "representative_pdb", "target", "best_pdb_id", "qtmscore",
        "alntmscore", "qcov", "tcov", "alnlen", "rmsd",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")

    if len(df) != 720:
        raise RuntimeError(f"Expected 720 model assignments for {peptide}; found {len(df)} in {path}")
    if set(df["peptide"].astype(str).str.lower()) != {peptide}:
        raise RuntimeError(f"Assignment table peptide labels are not exclusively {peptide}: {path}")

    # Stable sort with exact final ranking precedence.
    return df.sort_values(RANK_COLS, ascending=RANK_ASC, kind="mergesort").reset_index(drop=True)


def load_manifest(project: Path, peptide: str) -> pd.DataFrame:
    path = manifest_path(project, peptide)
    if not path.is_file():
        raise FileNotFoundError(f"Structure manifest not found: {path}")
    df = pd.read_csv(path, sep="\t")
    required = {"foldseek_id", "source_type", "source_file", "representative_pdb"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")
    df["foldseek_id_norm"] = df["foldseek_id"].map(normalize_id)
    return df


def build_selected_rows(project: Path) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []

    for peptide in ("ab40", "ab42"):
        df = load_assignments(project, peptide)

        # Per-method figures.
        for method in METHODS:
            top = df[df["method"] == method].head(5).copy()
            if len(top) != 5:
                raise RuntimeError(f"Expected 5 selected {method}/{peptide} models; found {len(top)}")
            top["selection_scope"] = method
            top["figure_group"] = f"{method}_{peptide}"
            top["selection_rank"] = np.arange(1, 6)
            blocks.append(top)

        # Overall top five across all four methods; no quota or balancing is imposed.
        # Therefore one method is allowed to occupy all five places if those truly are
        # the five highest qTM assignments.
        top_all = df.head(5).copy()
        top_all["selection_scope"] = "ALL"
        top_all["figure_group"] = f"ALL_{peptide}"
        top_all["selection_rank"] = np.arange(1, 6)
        blocks.append(top_all)

    selected = pd.concat(blocks, ignore_index=True)

    # Attach exact experimental representative paths from the peptide-specific manifest.
    exp_paths: dict[tuple[str, str], str] = {}
    for peptide in ("ab40", "ab42"):
        man = load_manifest(project, peptide)
        exp = man[man["source_type"] == "Experimental"].copy()
        for _, row in exp.iterrows():
            exp_paths[(peptide, row["foldseek_id_norm"])] = str(row["representative_pdb"])

    selected["experimental_representative_pdb"] = [
        exp_paths.get((pep, normalize_id(target)), "")
        for pep, target in zip(selected["peptide"], selected["target"])
    ]

    if (selected["experimental_representative_pdb"] == "").any():
        bad = selected.loc[selected["experimental_representative_pdb"] == "", ["peptide", "target"]]
        raise RuntimeError(
            "Could not resolve experimental representative paths for:\n" + bad.to_string(index=False)
        )

    # De-duplicate exact query-target pairs for rendering/contact computation; figure
    # membership is retained in the full selected table.
    return selected


# -----------------------------------------------------------------------------
# Contact Jaccard from exact representative chains
# -----------------------------------------------------------------------------

def read_ca_coordinates(pdb_path: Path) -> dict[tuple[int, str], np.ndarray]:
    """Read one representative-chain PDB's Cα coordinates keyed by (resseq, insertion code)."""
    coords: dict[tuple[int, str], np.ndarray] = {}
    if not pdb_path.is_file():
        raise FileNotFoundError(pdb_path)

    with pdb_path.open("r", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom = line[12:16].strip()
            if atom != "CA":
                continue
            altloc = line[16:17].strip()
            if altloc not in ("", "A", "1"):
                continue
            try:
                resseq = int(line[22:26])
                icode = line[26:27].strip()
                xyz = np.array([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ], dtype=float)
            except ValueError:
                continue
            key = (resseq, icode)
            if key not in coords:
                coords[key] = xyz

    if not coords:
        raise RuntimeError(f"No Cα coordinates found in {pdb_path}")
    return coords


def contact_set(coords: dict[tuple[int, str], np.ndarray]) -> set[tuple[tuple[int, str], tuple[int, str]]]:
    keys = sorted(coords, key=lambda x: (x[0], x[1]))
    contacts: set[tuple[tuple[int, str], tuple[int, str]]] = set()
    cutoff2 = CONTACT_CUTOFF_A ** 2

    for i in range(len(keys)):
        k1 = keys[i]
        for j in range(i + 1, len(keys)):
            k2 = keys[j]
            # Aβ representative chains use biological sequence numbering; insertion
            # codes are retained, but normal sequence-neighbour exclusion is numeric.
            if abs(k2[0] - k1[0]) < MIN_SEQUENCE_SEPARATION:
                continue
            delta = coords[k1] - coords[k2]
            if float(np.dot(delta, delta)) <= cutoff2:
                contacts.add((k1, k2))
    return contacts


def contact_metrics(pred_pdb: Path, exp_pdb: Path) -> dict[str, float | int]:
    pred = contact_set(read_ca_coordinates(pred_pdb))
    exp = contact_set(read_ca_coordinates(exp_pdb))
    intersection = pred & exp
    union = pred | exp

    jaccard = len(intersection) / len(union) if union else float("nan")
    pred_recovery = len(intersection) / len(pred) if pred else float("nan")
    exp_recovery = len(intersection) / len(exp) if exp else float("nan")

    return {
        "pred_contacts": len(pred),
        "exp_contacts": len(exp),
        "shared_contacts": len(intersection),
        "contact_jaccard": jaccard,
        "predicted_contact_recovery": pred_recovery,
        "experimental_contact_recovery": exp_recovery,
    }


# -----------------------------------------------------------------------------
# PyMOL rendering
# -----------------------------------------------------------------------------

def configure_pymol() -> None:
    from pymol import cmd

    cmd.reinitialize()
    cmd.bg_color("white")
    cmd.set("orthoscopic", 1)
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_shadows", 0)
    cmd.set("specular", 0.15)
    cmd.set("shininess", 10)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("cartoon_sampling", 14)
    cmd.set("cartoon_transparency", 0.0)


def render_overlay(pred_pdb: Path, exp_pdb: Path, out_png: Path, out_pse: Path | None = None) -> dict[str, float]:
    from pymol import cmd

    cmd.delete("all")
    cmd.load(str(exp_pdb), "experimental")
    cmd.load(str(pred_pdb), "prediction")
    cmd.remove("solvent")
    cmd.hide("everything", "all")
    cmd.show("cartoon", "experimental and polymer.protein")
    cmd.show("cartoon", "prediction and polymer.protein")
    cmd.color(EXP_COLOR, "experimental and polymer.protein")
    cmd.color(PRED_COLOR, "prediction and polymer.protein")

    # Structural superposition for visualisation only. Foldseek qTM/alnTM shown on
    # the panel are NOT recalculated here and remain the exact final Foldseek values.
    # cmd.super is structure-based and robust to the truncated experimental cores.
    result = cmd.super(
        "prediction and name CA",
        "experimental and name CA",
        cycles=5,
        transform=1,
        quiet=1,
    )

    cmd.orient("experimental and polymer.protein")
    cmd.zoom("experimental or prediction", buffer=3.5, complete=1)
    cmd.center("experimental or prediction")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd.png(
        str(out_png),
        width=PANEL_WIDTH,
        height=PANEL_HEIGHT,
        dpi=300,
        ray=1 if RAY_TRACE else 0,
        quiet=1,
    )

    if out_pse is not None:
        out_pse.parent.mkdir(parents=True, exist_ok=True)
        cmd.save(str(out_pse))

    # PyMOL super returns: RMS after refinement, atom count after refinement,
    # cycles, RMS before refinement, atom count before refinement, raw score, residues aligned
    try:
        return {
            "pymol_super_rmsd": float(result[0]),
            "pymol_super_atoms": int(result[1]),
            "pymol_super_cycles": int(result[2]),
            "pymol_super_rmsd_before": float(result[3]),
            "pymol_super_atoms_before": int(result[4]),
        }
    except Exception:
        return {
            "pymol_super_rmsd": float("nan"),
            "pymol_super_atoms": 0,
            "pymol_super_cycles": 0,
            "pymol_super_rmsd_before": float("nan"),
            "pymol_super_atoms_before": 0,
        }


# -----------------------------------------------------------------------------
# Figure assembly
# -----------------------------------------------------------------------------

def panel_title(row: pd.Series, letter: str, overall: bool) -> str:
    peptide = PEPTIDE_DISPLAY[row["peptide"]]
    method = METHOD_DISPLAY[row["method"]]
    templ = template_display(row["template_status"])
    model = int(row["model_rank"])
    mer = int(row["oligomer_size"])
    ref = str(row["best_pdb_id"]).upper()

    if overall:
        return (
            f"{letter}  {method}: {peptide} {mer}-mer, {templ}, model {model}\n"
            f"Nearest experimental reference: {ref}"
        )
    return (
        f"{letter}  {peptide} {mer}-mer, {templ}, model {model}\n"
        f"Nearest experimental reference: {ref}"
    )


def format_metric(value: float) -> str:
    return "—" if pd.isna(value) else f"{float(value):.3f}"


def assemble_figure(rows: pd.DataFrame, title: str, out_stem: Path, overall: bool) -> None:
    if len(rows) != 5:
        raise RuntimeError(f"Figure requires exactly 5 rows; got {len(rows)} for {title}")

    fig = plt.figure(figsize=(15, 9.5))
    grid = fig.add_gridspec(2, 6, hspace=0.23, wspace=0.08)

    for letter, (_, row), position in zip(LETTERS, rows.iterrows(), FIG_LAYOUT):
        ax = fig.add_subplot(grid[position[0], position[1]])
        image = Image.open(row["rendered_panel"]).convert("RGB")
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(panel_title(row, letter, overall), fontsize=10, fontweight="bold", pad=8)

        metrics = (
            f"qTM={format_metric(row['qtmscore'])}   "
            f"alnTM={format_metric(row['alntmscore'])}   "
            f"contact Jaccard={format_metric(row['contact_jaccard'])}"
        )
        ax.text(
            0.5, -0.025, metrics,
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
        )

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.015,
        "Model prediction: cyan    Experimental representative: salmon",
        ha="center", fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.95])

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_specs() -> list[tuple[int, str, str, str]]:
    return [
        (1, "AlphaFold", "ab40", "Five strongest AlphaFold 3 Aβ40 chain-level matches to experimental Aβ40 structures"),
        (2, "AlphaFold", "ab42", "Five strongest AlphaFold 3 Aβ42 chain-level matches to experimental Aβ42 structures"),
        (3, "Boltz", "ab40", "Five strongest Boltz-2 Aβ40 chain-level matches to experimental Aβ40 structures"),
        (4, "Boltz", "ab42", "Five strongest Boltz-2 Aβ42 chain-level matches to experimental Aβ42 structures"),
        (5, "Protenix", "ab40", "Five strongest Protenix-v2 Aβ40 chain-level matches to experimental Aβ40 structures"),
        (6, "Protenix", "ab42", "Five strongest Protenix-v2 Aβ42 chain-level matches to experimental Aβ42 structures"),
        (7, "OpenFold", "ab40", "Five strongest OpenFold3 Aβ40 chain-level matches to experimental Aβ40 structures"),
        (8, "OpenFold", "ab42", "Five strongest OpenFold3 Aβ42 chain-level matches to experimental Aβ42 structures"),
        (9, "ALL", "ab40", "Five strongest Aβ40 chain-level matches across all four prediction methods"),
        (10, "ALL", "ab42", "Five strongest Aβ42 chain-level matches across all four prediction methods"),
    ]


# -----------------------------------------------------------------------------
# Preflight / run
# -----------------------------------------------------------------------------

def print_selection_summary(selected: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("TOP-5 OVERLAY SELECTIONS (ranked by qTM, qcov, alnlen, RMSD)")
    print("=" * 100)

    for fig_no, scope, peptide, title in figure_specs():
        rows = selected[
            (selected["selection_scope"] == scope) & (selected["peptide"] == peptide)
        ].sort_values("selection_rank")
        print(f"\nFigure {fig_no}: {title}")
        for _, r in rows.iterrows():
            print(
                f"  {int(r['selection_rank'])}. {METHOD_DISPLAY[r['method']]} | "
                f"{int(r['oligomer_size'])}-mer | {template_display(r['template_status'])} | "
                f"model {int(r['model_rank'])} | {str(r['best_pdb_id']).upper()} | "
                f"qTM={r['qtmscore']:.4f} | alnTM={r['alntmscore']:.4f} | qcov={r['qcov']:.3f}"
            )


def preflight(project: Path, selected: pd.DataFrame) -> None:
    print("=" * 100)
    print("TOP-5 STRUCTURAL OVERLAY PREFLIGHT")
    print("=" * 100)
    print(f"Project: {project}")
    print(f"Selected figure rows: {len(selected)} (10 figures × 5 panels = 50 figure placements)")
    print(f"Unique prediction/reference pairs: {selected['foldseek_id'].nunique()}")

    missing: list[Path] = []
    for p in selected["representative_pdb"].astype(str):
        path = Path(p)
        if not path.is_file():
            missing.append(path)
    for p in selected["experimental_representative_pdb"].astype(str):
        path = Path(p)
        if not path.is_file():
            missing.append(path)

    if missing:
        print(f"\nMISSING STRUCTURE FILES: {len(missing)}")
        for p in missing[:30]:
            print(f"  {p}")
        if len(missing) > 30:
            print(f"  ... and {len(missing)-30} more")
        raise FileNotFoundError("Preflight failed because selected representative PDBs are missing.")

    # Verify PyMOL import, but do not render anything.
    try:
        import pymol  # noqa: F401
        print("PyMOL import: OK")
    except Exception as exc:
        raise RuntimeError(
            "PyMOL is not importable in the current Python environment.\n"
            "Activate the existing abeta_pymol environment and rerun.\n"
            f"Original error: {exc}"
        ) from exc

    print("Structure paths: ALL PRESENT")
    print_selection_summary(selected)
    print("\nPREFLIGHT PASSED. No source structures were modified.")


def run_all(project: Path, selected: pd.DataFrame, force: bool) -> None:
    out_analysis = output_analysis_dir(project)
    out_figs = output_figure_dir(project)
    panel_dir = out_figs / "rendered_panels"
    session_dir = out_figs / "pymol_sessions"
    out_analysis.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save a selection table before rendering, retaining figure membership.
    selection_path = out_analysis / "top5_overlay_selections.tsv"
    selected.to_csv(selection_path, sep="\t", index=False)

    configure_pymol()

    # Compute/render each unique query once. Overall figures reuse these panels.
    unique = selected.sort_values(["peptide", "method", "qtmscore"], ascending=[True, True, False])\
                     .drop_duplicates(subset=["foldseek_id"], keep="first")\
                     .copy()

    metric_rows: list[dict] = []
    panel_lookup: dict[str, str] = {}
    print("\n" + "=" * 100)
    print(f"CALCULATING CONTACT JACCARD + RENDERING {len(unique)} UNIQUE OVERLAYS")
    print("=" * 100)

    for idx, (_, row) in enumerate(unique.iterrows(), start=1):
        fid = str(row["foldseek_id"])
        pred = Path(row["representative_pdb"])
        exp = Path(row["experimental_representative_pdb"])
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", fid)
        target = clean_pdb_label(row["best_pdb_id"])
        out_png = panel_dir / f"{safe}_vs_{target}.png"
        out_pse = session_dir / f"{safe}_vs_{target}.pse"

        print(
            f"[{idx:02d}/{len(unique):02d}] {METHOD_DISPLAY[row['method']]} "
            f"{PEPTIDE_DISPLAY[row['peptide']]} {int(row['oligomer_size'])}-mer "
            f"{template_display(row['template_status'])} model {int(row['model_rank'])} vs {target}"
        )

        cm = contact_metrics(pred, exp)
        if force or not out_png.is_file() or out_png.stat().st_size == 0:
            pymol_metrics = render_overlay(pred, exp, out_png, out_pse)
        else:
            pymol_metrics = {
                "pymol_super_rmsd": float("nan"),
                "pymol_super_atoms": 0,
                "pymol_super_cycles": 0,
                "pymol_super_rmsd_before": float("nan"),
                "pymol_super_atoms_before": 0,
            }
            print("    existing rendered panel reused (use --force to re-render)")

        panel_lookup[fid] = str(out_png)
        metric_rows.append({
            "foldseek_id": fid,
            "method": row["method"],
            "peptide": row["peptide"],
            "template_status": row["template_status"],
            "oligomer_size": int(row["oligomer_size"]),
            "model_rank": int(row["model_rank"]),
            "best_pdb_id": str(row["best_pdb_id"]).upper(),
            "target": row["target"],
            "qtmscore": row["qtmscore"],
            "alntmscore": row["alntmscore"],
            "qcov": row["qcov"],
            "tcov": row["tcov"],
            "foldseek_rmsd": row["rmsd"],
            "alnlen": int(row["alnlen"]),
            "prediction_representative_pdb": str(pred),
            "experimental_representative_pdb": str(exp),
            "rendered_panel": str(out_png),
            **cm,
            **pymol_metrics,
        })

    metrics = pd.DataFrame(metric_rows)
    metrics_path = out_analysis / "top5_unique_pair_metrics.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)

    # Merge exact contact metrics and rendered panel into the 50 figure placements.
    plot_rows = selected.merge(
        metrics[["foldseek_id", "contact_jaccard", "pred_contacts", "exp_contacts",
                 "shared_contacts", "predicted_contact_recovery", "experimental_contact_recovery",
                 "pymol_super_rmsd", "rendered_panel"]],
        on="foldseek_id", how="left", validate="many_to_one",
    )
    plot_source = out_analysis / "top5_figure_source_data.tsv"
    plot_rows.to_csv(plot_source, sep="\t", index=False)

    print("\n" + "=" * 100)
    print("ASSEMBLING TEN FINAL FIGURES")
    print("=" * 100)

    for fig_no, scope, peptide, title in figure_specs():
        rows = plot_rows[
            (plot_rows["selection_scope"] == scope) & (plot_rows["peptide"] == peptide)
        ].sort_values("selection_rank")

        scope_name = "all_methods" if scope == "ALL" else scope.lower()
        stem = out_figs / f"figure_{fig_no:02d}_{peptide}_{scope_name}_top5_overlays"
        assemble_figure(rows, title, stem, overall=(scope == "ALL"))
        print(f"Figure {fig_no:02d}: {stem.with_suffix('.png')}")

    print("\n" + "=" * 100)
    print("TOP-5 STRUCTURAL OVERLAY ANALYSIS COMPLETE")
    print("=" * 100)
    print(f"Selection table: {selection_path}")
    print(f"Unique-pair metrics: {metrics_path}")
    print(f"Figure source data: {plot_source}")
    print(f"Rendered panels: {panel_dir}")
    print(f"PyMOL sessions: {session_dir}")
    print(f"Final figures: {out_figs}")
    print("Original prediction and experimental structures were not modified.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--preflight", action="store_true", help="Validate inputs and print the exact top-five selections.")
    parser.add_argument("--all", action="store_true", help="Compute contact Jaccard, render overlays, and make all ten figures.")
    parser.add_argument("--force", action="store_true", help="Re-render existing derived PyMOL panels.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project_root.resolve()
    selected = build_selected_rows(project)

    # No action flag deliberately means read-only preflight, mirroring the Foldseek workflow.
    if not args.preflight and not args.all:
        args.preflight = True

    if args.preflight:
        preflight(project, selected)
    if args.all:
        # --all includes the same validation first.
        preflight(project, selected)
        run_all(project, selected, args.force)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
