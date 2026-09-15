#!/usr/bin/env python3
"""
make_ecod_cath_master_figures.py

Create the two final ECOD/CATH-inspired master figures:

Aβ40:
  A. Experimentally defined structural taxonomy:
     dendrogram + metadata + experimental qTM matrix
  B. Prediction occupancy across experimental fold classes
  C. Prediction occupancy across experimental structural families
  D. Prediction occupancy across experimental polymorph groups

Aβ42:
  same layout and logic.

IMPORTANT
---------
This script DOES NOT recompute or alter:
- experimental structure clustering,
- metadata,
- Foldseek results,
- class/family/polymorph assignments,
- prediction-to-experimental assignments.

It only composes already-generated final V3/V4 figure panels into
clear dissertation-style master figures.

Default project:
    /home/benla/abeta_project_final

Outputs:
    <project>/figures/ecod_cath_hierarchy/master_figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


DEFAULT_PROJECT = Path("/home/benla/abeta_project_final")


def panel_paths(project: Path, peptide: str) -> dict[str, Path]:
    refined = project / "figures" / "ecod_cath_hierarchy" / "refinement_v4"
    projection = project / "figures" / "ecod_cath_hierarchy" / "prediction_projection"

    return {
        "A": refined / f"{peptide}_ecod_cath_integrated_qtm_metadata_refined.png",
        "B": projection / f"{peptide}_prediction_fold_class_occupancy.png",
        "C": projection / f"{peptide}_prediction_family_occupancy.png",
        "D": projection / f"{peptide}_prediction_polymorph_occupancy.png",
    }


def peptide_label(peptide: str) -> str:
    return "Aβ40" if peptide == "ab40" else "Aβ42"


def preflight(project: Path) -> bool:
    print("=" * 100)
    print("ECOD/CATH MASTER-FIGURE PREFLIGHT")
    print("=" * 100)
    print(f"Project: {project}")
    ok = True

    for peptide in ("ab40", "ab42"):
        print()
        print(peptide_label(peptide))
        for panel, path in panel_paths(project, peptide).items():
            if not path.is_file():
                ok = False
                print(f"  Panel {panel}: MISSING  {path}")
                continue
            with Image.open(path) as im:
                w, h = im.size
            print(f"  Panel {panel}: OK  {w}×{h}  {path}")

    outdir = project / "figures" / "ecod_cath_hierarchy" / "master_figures"
    print()
    print(f"Output directory: {outdir}")

    if ok:
        print()
        print("PREFLIGHT PASSED.")
        print("The experimental taxonomy is the reference system; prediction panels are projections onto it.")
    else:
        print()
        print("PREFLIGHT FAILED: one or more required final panels are missing.")

    return ok


def load_rgb(path: Path):
    with Image.open(path) as im:
        return im.convert("RGB").copy()


def make_master(project: Path, peptide: str, dpi: int = 300) -> tuple[Path, Path]:
    p = panel_paths(project, peptide)
    images = {k: load_rgb(v) for k, v in p.items()}

    label = peptide_label(peptide)
    outdir = project / "figures" / "ecod_cath_hierarchy" / "master_figures"
    outdir.mkdir(parents=True, exist_ok=True)

    # Large experimental taxonomy on the left; three projection panels stacked on the right.
    fig = plt.figure(figsize=(28, 18))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=(1.78, 1.0),
        height_ratios=(1, 1, 1),
        left=0.015,
        right=0.992,
        bottom=0.025,
        top=0.925,
        wspace=0.035,
        hspace=0.055,
    )

    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 1])
    axes = {"A": ax_a, "B": ax_b, "C": ax_c, "D": ax_d}

    for panel in ("A", "B", "C", "D"):
        ax = axes[panel]
        ax.imshow(images[panel], interpolation="none")
        ax.axis("off")
        x = -0.035 if panel == "A" else -0.065
        ax.text(
            x, 1.008, panel,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=24, fontweight="bold",
            clip_on=False,
        )

    fig.suptitle(
        f"{label}: experimentally defined hierarchical structural taxonomy "
        f"and projection of predicted models",
        fontsize=28,
        fontweight="bold",
        y=0.985,
    )

    # Explicit visual/semantic separation to prevent the prediction panels
    # being mistaken for inputs to the taxonomy.
    fig.text(
        0.305, 0.948,
        "EXPERIMENTALLY DEFINED REFERENCE TAXONOMY",
        ha="center", va="center",
        fontsize=18, fontweight="bold",
    )
    fig.text(
        0.820, 0.948,
        "PREDICTION PROJECTION ONTO EXPERIMENTAL TAXONOMY",
        ha="center", va="center",
        fontsize=18, fontweight="bold",
    )

    stem = f"{peptide}_ecod_cath_master_experimental_taxonomy_prediction_projection"
    png = outdir / f"{stem}.png"
    pdf = outdir / f"{stem}.pdf"

    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return png, pdf


def write_caption_file(project: Path) -> Path:
    outdir = project / "figures" / "ecod_cath_hierarchy" / "master_figures"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "MASTER_FIGURE_CAPTIONS.txt"

    text = """ECOD/CATH-INSPIRED MASTER FIGURE CAPTIONS
===========================================

Aβ40
----
Experimentally defined Aβ40 structural taxonomy and projection of predicted models.
(A) Hierarchical structural classification of the 33 experimentally determined Aβ40
representative chains, defined from exhaustive pairwise Foldseek qTM similarity and
annotated with curated experimental metadata. The nested hierarchy comprises five
broad fold classes, 11 structural families and 14 polymorph groups. Metadata are
annotations only and were not used to determine the clustering. (B-D) Occupancy of
the experimentally defined fold classes, structural families and polymorph groups,
respectively, by 720 Aβ40 predictions from AlphaFold 3, Boltz-2, OpenFold3 and
Protenix-v2, separated by template condition. Predicted structures were not included
when defining the experimental taxonomy; each prediction was projected onto the
hierarchy through its best experimental Foldseek match.

Aβ42
----
Experimentally defined Aβ42 structural taxonomy and projection of predicted models.
(A) Hierarchical structural classification of the 46 experimentally determined Aβ42
representative chains, defined from exhaustive pairwise Foldseek qTM similarity and
annotated with curated experimental metadata. The nested hierarchy comprises four
broad fold classes, 11 structural families and 21 polymorph groups. Metadata are
annotations only and were not used to determine the clustering. (B-D) Occupancy of
the experimentally defined fold classes, structural families and polymorph groups,
respectively, by 720 Aβ42 predictions from AlphaFold 3, Boltz-2, OpenFold3 and
Protenix-v2, separated by template condition. Predicted structures were not included
when defining the experimental taxonomy; each prediction was projected onto the
hierarchy through its best experimental Foldseek match.

Suggested Methods wording
-------------------------
An ECOD/CATH-inspired hierarchical structural taxonomy was constructed independently
for Aβ40 and Aβ42 from experimentally determined representative chains only.
Directional query-normalised Foldseek TM-scores were symmetrised for each experimental
pair, converted to structural distances (1 - symmetric qTM), and clustered by
average-linkage hierarchical clustering. Nested cut levels defined broad fold classes,
structural families and polymorph groups. Experimental metadata were added only after
structural classification and did not contribute to clustering. Predicted structures
from AlphaFold 3, Boltz-2, OpenFold3 and Protenix-v2 were subsequently projected onto
the fixed experimental hierarchy according to their highest-qTM experimental match.
"""
    path.write_text(text, encoding="utf-8")
    return path


def write_manifest(project: Path) -> Path:
    outdir = project / "figures" / "ecod_cath_hierarchy" / "master_figures"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "MASTER_FIGURE_PANEL_MANIFEST.tsv"
    lines = ["peptide\tpanel\tmeaning\tsource_file"]
    meanings = {
        "A": "Experimentally defined taxonomy: dendrogram + metadata + experimental qTM matrix",
        "B": "Prediction occupancy across experimentally defined fold classes",
        "C": "Prediction occupancy across experimentally defined structural families",
        "D": "Prediction occupancy across experimentally defined polymorph groups",
    }
    for peptide in ("ab40", "ab42"):
        for panel, source in panel_paths(project, peptide).items():
            lines.append(
                f"{peptide}\t{panel}\t{meanings[panel]}\t{source}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT,
        help=f"Project root (default: {DEFAULT_PROJECT})",
    )
    ap.add_argument(
        "--preflight",
        action="store_true",
        help="Check required final panels only; create nothing.",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Create both final master figures, captions and panel manifest.",
    )
    ap.add_argument("--dpi", type=int, default=300)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project_root.resolve()

    if args.preflight or not args.all:
        ok = preflight(project)
        if not ok:
            return 1
        if args.preflight and not args.all:
            return 0

    if args.all:
        if not preflight(project):
            return 1

        print()
        print("=" * 100)
        print("CREATING FINAL MASTER FIGURES")
        print("=" * 100)

        for peptide in ("ab40", "ab42"):
            png, pdf = make_master(project, peptide, dpi=args.dpi)
            print(f"{peptide_label(peptide)} PNG: {png}")
            print(f"{peptide_label(peptide)} PDF: {pdf}")

        captions = write_caption_file(project)
        manifest = write_manifest(project)
        print(f"Captions: {captions}")
        print(f"Manifest: {manifest}")
        print()
        print("MASTER FIGURES COMPLETE.")
        print("No experimental taxonomy, metadata, Foldseek result or prediction assignment was modified.")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
