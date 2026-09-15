"""
Create four publication-style PyMOL montage figures for the AF3 Aβ project.

Each figure contains five AF3 models (columns) for 18 oligomer sizes (rows):
1–15mer, 20mer, 30mer and 50mer. Structures are coloured by atomic pLDDT
stored in the B-factor field using the exact requested PyMOL spectrum:

    spectrum b, red_yellow_green_cyan_blue, minimum=50, maximum=90

Recommended Ubuntu/WSL setup:
    conda create -n abeta_pymol -c conda-forge python=3.11 pymol-open-source pillow
    conda activate abeta_pymol
    python make_abeta_pymol_montages.py

The four final PNG files are placed in AF3/pymol_montage_figures.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Windows location supplied by Ben. The script converts this automatically to
# /mnt/c/... when run inside Ubuntu/WSL. Change only this value if AF3 is moved.
WINDOWS_AF3_ROOT = Path(
    r"C:\Users\benla\OneDrive\Documents\Bioinformatics_MSc\LIFE703\Data\AF3"
)

MERS = list(range(1, 16)) + [20, 30, 50]
MODELS = list(range(5))
CONDITIONS = (
    ("ab40", "with_templates", "Aβ40 – with templates"),
    ("ab40", "no_templates", "Aβ40 – without templates"),
    ("ab42", "with_templates", "Aβ42 – with templates"),
    ("ab42", "no_templates", "Aβ42 – without templates"),
)

# Individual PyMOL render size. Increase these for still higher-resolution output.
TILE_WIDTH = 620
TILE_HEIGHT = 430
RAY_TRACE = True

# Montage spacing and labelling.
LEFT_MARGIN = 135
TOP_MARGIN = 150
RIGHT_MARGIN = 35
BOTTOM_MARGIN = 40
ROW_GAP = 8
COLUMN_GAP = 8
BACKGROUND = "white"

MER_RE = re.compile(r"_(\d+)mer_", re.IGNORECASE)
MODEL_RE = re.compile(r"_model_(\d+)\.(?:cif|pdb)$", re.IGNORECASE)


def windows_path_to_wsl(path: Path) -> Path:
    """Convert C:\\folder\\file to /mnt/c/folder/file when appropriate."""
    text = str(path)
    match = re.match(r"^([A-Za-z]):\\(.*)$", text)
    if match and sys.platform.startswith("linux"):
        drive, remainder = match.groups()
        return Path("/mnt") / drive.lower() / Path(remainder.replace("\\", "/"))
    return path


def resolve_af3_root() -> Path:
    """Use an optional command-line root, otherwise use the configured path."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()
    return windows_path_to_wsl(WINDOWS_AF3_ROOT)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a clean Linux/Windows font, with a safe Pillow fallback."""
    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def centred_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill="black"):
    """Draw text horizontally and vertically centred on xy."""
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def locate_structure_files(condition_dir: Path) -> dict[tuple[int, int], Path]:
    """Locate each mer/model structure robustly from AF3 folder/file names."""
    found: dict[tuple[int, int], Path] = {}

    for structure in condition_dir.rglob("*"):
        if not structure.is_file() or structure.suffix.lower() not in {".cif", ".pdb"}:
            continue

        model_match = MODEL_RE.search(structure.name)
        if not model_match:
            continue

        mer_match = None
        # Usually both folder and filename contain the mer; checking parents makes
        # the script tolerant of renamed structure files.
        for candidate in (structure.name, structure.parent.name):
            mer_match = MER_RE.search(candidate)
            if mer_match:
                break

        if not mer_match:
            continue

        key = (int(mer_match.group(1)), int(model_match.group(1)))
        if key in found:
            raise RuntimeError(
                f"Duplicate structure for {key[0]}mer model {key[1]}:\n"
                f"  {found[key]}\n  {structure}"
            )
        found[key] = structure

    expected = {(mer, model) for mer in MERS for model in MODELS}
    missing = sorted(expected - set(found))
    extras = sorted(set(found) - expected)

    if missing:
        description = ", ".join(f"{mer}mer/model_{model}" for mer, model in missing)
        raise FileNotFoundError(
            f"Missing {len(missing)} expected structure file(s) in {condition_dir}:\n"
            f"{description}"
        )
    if extras:
        print(f"  Note: ignoring {len(extras)} structure(s) outside the requested set")

    return {key: found[key] for key in sorted(expected)}


def configure_pymol() -> None:
    """Set a consistent, clean, white-background PyMOL rendering style."""
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


def render_structure(structure: Path, output_png: Path) -> None:
    """Render one model as a pLDDT-coloured cartoon."""
    from pymol import cmd

    cmd.delete("all")
    cmd.load(str(structure), "model")
    cmd.remove("solvent")
    cmd.hide("everything", "all")
    cmd.show("cartoon", "polymer.protein")

    # Exact colouring requested by Ben. AF3 writes pLDDT to the B-factor field.
    cmd.spectrum(
        "b",
        "red_yellow_green_cyan_blue",
        "polymer.protein",
        minimum=50,
        maximum=90,
    )

    cmd.orient("polymer.protein")
    cmd.zoom("polymer.protein", buffer=2.5, complete=1)
    cmd.center("polymer.protein")
    cmd.png(
        str(output_png),
        width=TILE_WIDTH,
        height=TILE_HEIGHT,
        dpi=300,
        ray=1 if RAY_TRACE else 0,
        quiet=1,
    )


def render_condition(
    condition_dir: Path,
    structures: dict[tuple[int, int], Path],
    render_dir: Path,
) -> dict[tuple[int, int], Path]:
    """Create the 90 individual transparent-safe tiles for one condition."""
    render_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[tuple[int, int], Path] = {}

    for index, ((mer, model), structure) in enumerate(structures.items(), start=1):
        output_png = render_dir / f"{mer:02d}mer_model_{model}.png"
        print(f"    [{index:02d}/90] {mer}mer model_{model}")
        render_structure(structure, output_png)
        if not output_png.is_file() or output_png.stat().st_size == 0:
            raise RuntimeError(f"PyMOL did not create a valid image: {output_png}")
        rendered[(mer, model)] = output_png

    return rendered


def make_montage(
    rendered: dict[tuple[int, int], Path],
    title: str,
    output_png: Path,
) -> None:
    """Assemble one labelled 5-column × 18-row figure like the supplied example."""
    width = LEFT_MARGIN + 5 * TILE_WIDTH + 4 * COLUMN_GAP + RIGHT_MARGIN
    height = TOP_MARGIN + 18 * TILE_HEIGHT + 17 * ROW_GAP + BOTTOM_MARGIN
    montage = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(montage)

    title_font = load_font(42, bold=True)
    column_font = load_font(26, bold=True)
    row_font = load_font(26, bold=True)
    note_font = load_font(20, bold=False)

    centred_text(draw, (width // 2, 36), title, title_font)
    centred_text(
        draw,
        (width // 2, 82),
        "Cartoon representation coloured by pLDDT (red ≤50; blue ≥90)",
        note_font,
        fill=(55, 55, 55),
    )

    for column, model in enumerate(MODELS):
        x = LEFT_MARGIN + column * (TILE_WIDTH + COLUMN_GAP) + TILE_WIDTH // 2
        centred_text(draw, (x, 122), f"AF3 model_{model}", column_font)

    for row, mer in enumerate(MERS):
        y = TOP_MARGIN + row * (TILE_HEIGHT + ROW_GAP)
        centred_text(draw, (LEFT_MARGIN // 2, y + TILE_HEIGHT // 2), f"{mer}mer", row_font)

        for column, model in enumerate(MODELS):
            x = LEFT_MARGIN + column * (TILE_WIDTH + COLUMN_GAP)
            with Image.open(rendered[(mer, model)]) as tile:
                tile = tile.convert("RGB")
                montage.paste(tile, (x, y))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output_png, dpi=(300, 300), optimize=True)


def main() -> None:
    af3_root = resolve_af3_root()
    if not af3_root.is_dir():
        raise FileNotFoundError(
            f"AF3 root directory was not found:\n  {af3_root}\n\n"
            "If it has moved, pass its location after the script name, for example:\n"
            "  python make_abeta_pymol_montages.py /mnt/c/path/to/AF3"
        )

    try:
        import pymol
        from pymol import cmd
    except ImportError as error:
        raise SystemExit(
            "PyMOL's Python module is not installed in this environment.\n"
            "Install it with:\n"
            "  conda install -c conda-forge pymol-open-source pillow"
        ) from error

    pymol.finish_launching(["pymol", "-cq"])
    output_dir = af3_root / "pymol_montage_figures"
    tile_root = output_dir / "individual_model_renders"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"AF3 root: {af3_root}")
    print("Validating all expected structures before rendering...")
    validated = {}
    for peptide, template_folder, title in CONDITIONS:
        condition_dir = af3_root / peptide / template_folder
        if not condition_dir.is_dir():
            raise FileNotFoundError(f"Condition directory not found: {condition_dir}")
        validated[(peptide, template_folder)] = locate_structure_files(condition_dir)
        print(f"  {title}: 90/90 structures found")

    configure_pymol()
    for condition_number, (peptide, template_folder, title) in enumerate(CONDITIONS, start=1):
        condition_key = f"{peptide}_{template_folder}"
        print(f"\n[{condition_number}/4] Rendering {title}")
        render_dir = tile_root / condition_key
        rendered = render_condition(
            af3_root / peptide / template_folder,
            validated[(peptide, template_folder)],
            render_dir,
        )
        final_png = output_dir / f"{condition_key}_all_models_plddt.png"
        make_montage(rendered, title, final_png)
        print(f"  Finished: {final_png}")

    cmd.quit()
    print("\nAll four montage figures were created successfully:")
    for peptide, template_folder, _ in CONDITIONS:
        print(f"  {output_dir / f'{peptide}_{template_folder}_all_models_plddt.png'}")


if __name__ == "__main__":
    main()
