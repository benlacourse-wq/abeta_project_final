#!/usr/bin/env python3

"""
Create 12 PyMOL pLDDT montage figures from the completed ABCFold dataset.

Predictors:
    Boltz-2
    OpenFold3
    Protenix

Conditions:
    Aβ40 without templates
    Aβ40 with templates
    Aβ42 without templates
    Aβ42 with templates

Each figure:
    18 oligomer sizes:
        1-15mer, 20mer, 30mer, 50mer

    5 models:
        model_0 ... model_4

The script uses the canonical ABCFold output_models CIF files.

No source files are altered.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# PATHS
# ============================================================

RESULTS_ROOT = Path(
    "/home/benla/abeta_project_final/ABC_Fold/"
    "ab40_ab42_results"
)

OUTPUT_ROOT = Path(
    "/home/benla/abeta_project_final/ABC_Fold/"
    "pymol_predictor_montage_figures"
)


# ============================================================
# DATASETS
# ============================================================

MERS = list(range(1, 16)) + [20, 30, 50]
MODELS = list(range(5))

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


# ============================================================
# MONTAGE DIMENSIONS
# Same basic arrangement as your AF3 figure
# ============================================================

TILE_WIDTH = 620
TILE_HEIGHT = 430

LEFT_MARGIN = 135
TOP_MARGIN = 150
RIGHT_MARGIN = 35
BOTTOM_MARGIN = 40

ROW_GAP = 8
COLUMN_GAP = 8

BACKGROUND = "white"


# ============================================================
# ABCFOLD PATHS
# ============================================================

def condition_directory(
    peptide,
    template_mode,
    mer,
):

    peptide_upper = peptide.upper()

    if template_mode == "no_templates":

        group = (
            RESULTS_ROOT
            / f"ABCFold_{peptide_upper}_no_templates"
        )

        condition = (
            group
            / f"{peptide}_{mer}mer_no_templates"
        )

    else:

        group = (
            RESULTS_ROOT
            / f"ABCFold_{peptide_upper}_templates"
        )

        condition = (
            group
            / f"{peptide}_{mer}mer_with_templates"
        )

    return condition


def structure_path(
    predictor,
    peptide,
    template_mode,
    mer,
    model,
):

    return (
        condition_directory(
            peptide,
            template_mode,
            mer,
        )
        / "output_models"
        / f"{predictor}_model_{model}.cif"
    )


# ============================================================
# FONT HELPERS
# ============================================================

def load_font(size, bold=False):

    if bold:

        candidates = [
            "/usr/share/fonts/truetype/dejavu/"
            "DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]

    else:

        candidates = [
            "/usr/share/fonts/truetype/dejavu/"
            "DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]

    for candidate in candidates:

        if Path(candidate).is_file():

            return ImageFont.truetype(
                candidate,
                size=size,
            )

    return ImageFont.load_default()


def centred_text(
    draw,
    xy,
    text,
    font,
    fill="black",
):

    box = draw.textbbox(
        (0, 0),
        text,
        font=font,
    )

    width = box[2] - box[0]
    height = box[3] - box[1]

    draw.text(
        (
            xy[0] - width / 2,
            xy[1] - height / 2,
        ),
        text,
        font=font,
        fill=fill,
    )


# ============================================================
# PREFLIGHT
# ============================================================

def preflight(predictors):

    print("=" * 78)
    print("ABCFOLD PYMOL MONTAGE PREFLIGHT")
    print("=" * 78)

    print(f"Results root: {RESULTS_ROOT}")
    print()

    missing = []

    expected = (
        len(predictors)
        * len(CONDITIONS)
        * len(MERS)
        * len(MODELS)
    )

    found = 0

    for predictor in predictors:

        predictor_label = PREDICTORS[predictor]

        for (
            peptide,
            template_mode,
            peptide_label,
            template_label,
        ) in CONDITIONS:

            condition_found = 0

            for mer in MERS:

                for model in MODELS:

                    path = structure_path(
                        predictor,
                        peptide,
                        template_mode,
                        mer,
                        model,
                    )

                    if path.is_file():

                        found += 1
                        condition_found += 1

                    else:

                        missing.append(path)

            print(
                f"{predictor_label:<9} | "
                f"{peptide_label:<4} | "
                f"{template_label:<17} "
                f"{condition_found}/90"
            )

    print()
    print(f"Expected structures: {expected}")
    print(f"Found:               {found}")
    print(f"Missing:             {len(missing)}")

    if missing:

        print()
        print("MISSING FILES")
        print("-------------")

        for path in missing:

            print(path)

        raise FileNotFoundError(
            f"{len(missing)} required structures "
            "are missing."
        )

    print()
    print(
        "Preflight passed: every requested "
        "structure is present."
    )


# ============================================================
# PYMOL SETUP
# ============================================================

def configure_pymol():

    from pymol import cmd

    cmd.reinitialize()

    cmd.bg_color("white")

    cmd.set("orthoscopic", 1)

    cmd.set(
        "ray_opaque_background",
        1,
    )

    cmd.set("antialias", 2)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_shadows", 0)

    cmd.set("specular", 0.15)
    cmd.set("shininess", 10)

    cmd.set(
        "cartoon_fancy_helices",
        1,
    )

    cmd.set(
        "cartoon_smooth_loops",
        1,
    )

    cmd.set(
        "cartoon_sampling",
        14,
    )


# ============================================================
# INDIVIDUAL STRUCTURE RENDER
# ============================================================

def render_structure(
    structure,
    output_png,
    ray_trace=True,
):

    from pymol import cmd

    cmd.delete("all")

    cmd.load(
        str(structure),
        "model",
    )

    cmd.remove("solvent")

    cmd.hide(
        "everything",
        "all",
    )

    cmd.show(
        "cartoon",
        "polymer.protein",
    )

    # --------------------------------------------------------
    # Same pLDDT colouring as the AF3 montage
    # --------------------------------------------------------

    cmd.spectrum(
        "b",
        "red_yellow_green_cyan_blue",
        "polymer.protein",
        minimum=50,
        maximum=90,
    )

    cmd.orient(
        "polymer.protein",
    )

    cmd.zoom(
        "polymer.protein",
        buffer=2.5,
        complete=1,
    )

    cmd.center(
        "polymer.protein",
    )

    output_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cmd.png(
        str(output_png),
        width=TILE_WIDTH,
        height=TILE_HEIGHT,
        dpi=300,
        ray=1 if ray_trace else 0,
        quiet=1,
    )


# ============================================================
# RENDER ONE 90-STRUCTURE DATASET
# ============================================================

def render_dataset(
    predictor,
    peptide,
    template_mode,
    render_dir,
    ray_trace,
    force,
):

    rendered = {}

    render_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    counter = 0

    for mer in MERS:

        for model in MODELS:

            counter += 1

            structure = structure_path(
                predictor,
                peptide,
                template_mode,
                mer,
                model,
            )

            output_png = (
                render_dir
                / f"{mer:02d}mer_model_{model}.png"
            )

            # Makes the script resumable.
            if (
                not force
                and output_png.is_file()
                and output_png.stat().st_size > 0
            ):

                print(
                    f"  [{counter:02d}/90] "
                    f"{mer:>2}mer model_{model} "
                    "(reusing existing render)"
                )

            else:

                print(
                    f"  [{counter:02d}/90] "
                    f"{mer:>2}mer model_{model}"
                )

                render_structure(
                    structure,
                    output_png,
                    ray_trace,
                )

            if (
                not output_png.is_file()
                or output_png.stat().st_size == 0
            ):

                raise RuntimeError(
                    "PyMOL failed to create:\n"
                    f"{output_png}"
                )

            rendered[
                (mer, model)
            ] = output_png

    return rendered


# ============================================================
# MONTAGE
# ============================================================

def make_montage(
    rendered,
    predictor_label,
    peptide_label,
    template_label,
    output_png,
):

    width = (
        LEFT_MARGIN
        + 5 * TILE_WIDTH
        + 4 * COLUMN_GAP
        + RIGHT_MARGIN
    )

    height = (
        TOP_MARGIN
        + 18 * TILE_HEIGHT
        + 17 * ROW_GAP
        + BOTTOM_MARGIN
    )

    montage = Image.new(
        "RGB",
        (width, height),
        BACKGROUND,
    )

    draw = ImageDraw.Draw(
        montage,
    )

    title_font = load_font(
        42,
        bold=True,
    )

    column_font = load_font(
        26,
        bold=True,
    )

    row_font = load_font(
        26,
        bold=True,
    )

    note_font = load_font(
        20,
    )

    title = (
        f"{peptide_label} – "
        f"{template_label} – "
        f"{predictor_label}"
    )

    centred_text(
        draw,
        (width // 2, 36),
        title,
        title_font,
    )

    centred_text(
        draw,
        (width // 2, 82),
        (
            "Cartoon representation coloured by "
            "pLDDT (red ≤50; blue ≥90)"
        ),
        note_font,
        fill=(55, 55, 55),
    )

    # --------------------------------------------------------
    # COLUMN LABELS
    # --------------------------------------------------------

    for column, model in enumerate(MODELS):

        x = (
            LEFT_MARGIN
            + column
            * (TILE_WIDTH + COLUMN_GAP)
            + TILE_WIDTH // 2
        )

        centred_text(
            draw,
            (x, 122),
            (
                f"{predictor_label} "
                f"model_{model}"
            ),
            column_font,
        )

    # --------------------------------------------------------
    # ROWS
    # --------------------------------------------------------

    for row, mer in enumerate(MERS):

        y = (
            TOP_MARGIN
            + row
            * (TILE_HEIGHT + ROW_GAP)
        )

        centred_text(
            draw,
            (
                LEFT_MARGIN // 2,
                y + TILE_HEIGHT // 2,
            ),
            f"{mer}mer",
            row_font,
        )

        for column, model in enumerate(MODELS):

            x = (
                LEFT_MARGIN
                + column
                * (TILE_WIDTH + COLUMN_GAP)
            )

            tile_path = rendered[
                (mer, model)
            ]

            with Image.open(
                tile_path
            ) as tile:

                tile = tile.convert("RGB")

                montage.paste(
                    tile,
                    (x, y),
                )

    output_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    montage.save(
        output_png,
        dpi=(300, 300),
        optimize=True,
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictor",
        choices=[
            "all",
            "boltz",
            "openfold",
            "protenix",
        ],
        default="all",
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )

    parser.add_argument(
        "--no-ray",
        action="store_true",
        help=(
            "Disable ray tracing for a "
            "faster test run."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-render tiles even if "
            "they already exist."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    if not RESULTS_ROOT.is_dir():

        raise FileNotFoundError(
            "ABCFold results directory "
            f"not found:\n{RESULTS_ROOT}"
        )

    if args.predictor == "all":

        predictors = list(
            PREDICTORS.keys()
        )

    else:

        predictors = [
            args.predictor
        ]

    preflight(
        predictors,
    )

    if args.preflight_only:

        print()
        print(
            "Preflight-only mode complete. "
            "No images were created."
        )

        return

    try:

        import pymol
        from pymol import cmd

    except ImportError as error:

        raise SystemExit(
            "PyMOL is not installed in this "
            "Python environment.\n\n"
            "Use:\n"
            "  conda activate abeta_pymol"
        ) from error

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    tile_root = (
        OUTPUT_ROOT
        / "individual_model_renders"
    )

    pymol.finish_launching(
        ["pymol", "-cq"]
    )

    configure_pymol()

    total_datasets = (
        len(predictors)
        * len(CONDITIONS)
    )

    dataset_number = 0
    final_figures = []

    for predictor in predictors:

        predictor_label = (
            PREDICTORS[predictor]
        )

        for (
            peptide,
            template_mode,
            peptide_label,
            template_label,
        ) in CONDITIONS:

            dataset_number += 1

            print()
            print("=" * 78)

            print(
                f"[{dataset_number}/"
                f"{total_datasets}] "
                f"{predictor_label}: "
                f"{peptide_label}, "
                f"{template_label}"
            )

            print("=" * 78)

            condition_key = (
                f"{predictor}_"
                f"{peptide}_"
                f"{template_mode}"
            )

            render_dir = (
                tile_root
                / condition_key
            )

            rendered = render_dataset(
                predictor,
                peptide,
                template_mode,
                render_dir,
                ray_trace=not args.no_ray,
                force=args.force,
            )

            final_png = (
                OUTPUT_ROOT
                / (
                    f"{condition_key}"
                    "_all_models_plddt.png"
                )
            )

            make_montage(
                rendered,
                predictor_label,
                peptide_label,
                template_label,
                final_png,
            )

            final_figures.append(
                final_png
            )

            print()
            print(
                f"Finished: {final_png}"
            )

    cmd.quit()

    print()
    print("=" * 78)
    print("ALL MONTAGES FINISHED")
    print("=" * 78)

    print(
        f"Figures created: "
        f"{len(final_figures)}"
    )

    print(
        f"Models represented: "
        f"{len(final_figures) * 90}"
    )

    print()
    print(
        f"Output folder:\n"
        f"{OUTPUT_ROOT}"
    )

    print()

    for path in final_figures:

        print(path)


if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print(
            f"\nERROR: {error}",
            file=sys.stderr,
        )

        sys.exit(1)