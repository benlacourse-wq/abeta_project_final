#!/usr/bin/env python3

from pathlib import Path
import shutil
import csv
import sys

OLD = Path("/home/benla/abeta_project")
PROJECT = Path("/home/benla/abeta_project_final")
NEW = (
    PROJECT
    / "raw_predictions"
    / "ABC_Fold"
    / "ab40_ab42_results"
)

MANIFEST = PROJECT / "manifests" / "legacy_gapfill_manifest.tsv"

# ---------------------------------------------------------------------
# ONLY THESE THREE CONDITIONS ARE BEING RECOVERED.
#
# 20mer and 50mer gaps are intentionally untouched because equivalent
# old models do not exist.
# ---------------------------------------------------------------------

CASES = [
    {
        "peptide": "ab40",
        "PEPTIDE": "AB40",
        "mer": 5,
        "template": "no_templates",
        "condition_dir": "ABCFold_AB40_no_templates",
        "new_seed": "1638627401",

        "boltz_source": (
            OLD
            / "comparative_modelling/direct_boltz/outputs"
            / "ab40/no_templates/ab40_5mer_no_templates"
            / "boltz_results_ab40_5mer_no_templates"
            / "predictions/ab40_5mer_no_templates"
        ),
        "boltz_prefix": "ab40_5mer_no_templates",

        "openfold_source": (
            OLD
            / "openfold3_abeta/openfold3_abeta/openfold3_production_300"
            / "ab40/no_templates/5mer"
            / "openfold_ab40_5mer_no_templates"
            / "openfold_results_seed-1"
            / "ab40_5mer_no_templates/seed_1"
        ),
        "openfold_prefix": "ab40_5mer_no_templates_seed_1",

        "protenix_source": (
            OLD
            / "protenix/AB40/No_Templates"
            / "AB40_5mer_No_Templates"
            / "AB40_5mer_No_Templates"
            / "seed_68224/predictions"
        ),
        "protenix_prefix": "AB40_5mer_No_Templates",
    },

    {
        "peptide": "ab40",
        "PEPTIDE": "AB40",
        "mer": 5,
        "template": "with_templates",
        "condition_dir": "ABCFold_AB40_templates",
        "new_seed": "1638627401",

        # This is the repaired old Aβ40 5mer template-enabled Boltz run.
        "boltz_source": (
            OLD
            / "comparative_modelling/direct_boltz/outputs"
            / "ab40/with_templates/ab40_5mer_with_templates"
            / "boltz_results_ab40_5mer_one_template_test"
            / "predictions/ab40_5mer_one_template_test"
        ),
        "boltz_prefix": "ab40_5mer_one_template_test",

        "openfold_source": (
            OLD
            / "openfold3_abeta/openfold3_abeta/openfold3_production_300"
            / "ab40/with_templates/5mer"
            / "openfold_ab40_5mer_with_templates"
            / "openfold_results_seed-1"
            / "ab40_5mer_with_templates/seed_1"
        ),
        "openfold_prefix": "ab40_5mer_with_templates_seed_1",

        "protenix_source": (
            OLD
            / "protenix/AB40/Templates"
            / "AB40_5mer_Templates"
            / "AB40_5mer_Templates"
            / "seed_68224/predictions"
        ),
        "protenix_prefix": "AB40_5mer_Templates",
    },

    {
        "peptide": "ab42",
        "PEPTIDE": "AB42",
        "mer": 1,
        "template": "with_templates",
        "condition_dir": "ABCFold_AB42_templates",
        "new_seed": "1478292162",

        "boltz_source": (
            OLD
            / "comparative_modelling/direct_boltz/outputs"
            / "ab42/with_templates/ab42_1mer_with_templates"
            / "boltz_results_ab42_1mer_with_templates"
            / "predictions/ab42_1mer_with_templates"
        ),
        "boltz_prefix": "ab42_1mer_with_templates",

        "openfold_source": (
            OLD
            / "openfold3_abeta/openfold3_abeta/openfold3_production_300"
            / "ab42/with_templates/1mer"
            / "openfold_ab42_1mer_with_templates"
            / "openfold_results_seed-1"
            / "ab42_1mer_with_templates/seed_1"
        ),
        "openfold_prefix": "ab42_1mer_with_templates_seed_1",

        "protenix_source": (
            OLD
            / "protenix/AB42/Templates"
            / "AB42_1mer_Templates"
            / "AB42_1mer_Templates"
            / "seed_68224/predictions"
        ),
        "protenix_prefix": "AB42_1mer_Templates",
    },
]


manifest_rows = []


def checked_copy(src: Path, dst: Path, case, predictor, kind):
    """Copy without ever overwriting an existing genuine file."""

    if not src.is_file():
        raise FileNotFoundError(f"Required source file missing:\n{src}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing file:\n"
            f"{dst}\n"
            "This script only fills genuinely missing results."
        )

    shutil.copy2(src, dst)

    manifest_rows.append(
        {
            "predictor": predictor,
            "peptide": case["peptide"],
            "template_condition": case["template"],
            "oligomer_size": case["mer"],
            "file_type": kind,
            "source_file": str(src),
            "destination_file": str(dst),
            "provenance": "legacy_gapfill_from_abeta_project",
        }
    )


def recover_boltz(case, condition):
    pep = case["peptide"]
    PEP = case["PEPTIDE"]
    mer = case["mer"]
    seed = case["new_seed"]

    old_dir = case["boltz_source"]
    old_prefix = case["boltz_prefix"]

    new_prefix = f"abc_{pep}_{mer}mer_mmseqs_seed-{seed}"

    pred_dir = (
        condition
        / f"boltz_{PEP}_{mer}mer"
        / f"boltz_results_{new_prefix}"
        / "predictions"
        / new_prefix
    )

    output_dir = condition / "output_models"

    for i in range(5):

        # Coordinates
        src = old_dir / f"{old_prefix}_model_{i}.cif"
        dst = pred_dir / f"{new_prefix}_model_{i}.cif"

        checked_copy(src, dst, case, "Boltz", "model_cif")

        # ABCFold's convenient consolidated coordinate directory
        checked_copy(
            src,
            output_dir / f"boltz_model_{i}.cif",
            case,
            "Boltz",
            "output_model_cif",
        )

        # Confidence JSON
        checked_copy(
            old_dir / f"confidence_{old_prefix}_model_{i}.json",
            pred_dir / f"confidence_{new_prefix}_model_{i}.json",
            case,
            "Boltz",
            "confidence_json",
        )

        # Retain Boltz confidence arrays needed later for PAE/pLDDT work.
        for metric in ("pae", "pde", "plddt"):
            source = old_dir / f"{metric}_{old_prefix}_model_{i}.npz"

            if source.is_file():
                checked_copy(
                    source,
                    pred_dir / f"{metric}_{new_prefix}_model_{i}.npz",
                    case,
                    "Boltz",
                    f"{metric}_npz",
                )


def recover_openfold(case, condition):
    PEP = case["PEPTIDE"]
    mer = case["mer"]
    seed = case["new_seed"]

    old_dir = case["openfold_source"]
    old_prefix = case["openfold_prefix"]

    new_prefix = f"{PEP}_{mer}mer_seed_{seed}"

    pred_dir = (
        condition
        / f"openfold_{PEP}_{mer}mer"
        / f"openfold_results_seed-{seed}"
        / f"{PEP}_{mer}mer"
        / f"seed_{seed}"
    )

    output_dir = condition / "output_models"

    # OpenFold numbering is sample 1...5.
    # ABCFold output_models uses model 0...4.
    for sample in range(1, 6):

        model_index = sample - 1

        raw_src = old_dir / f"{old_prefix}_sample_{sample}_model.cif"
        fixed_src = old_dir / f"{old_prefix}_sample_{sample}_model_fixed.cif"

        checked_copy(
            raw_src,
            pred_dir / f"{new_prefix}_sample_{sample}_model.cif",
            case,
            "OpenFold",
            "model_cif",
        )

        checked_copy(
            fixed_src,
            pred_dir / f"{new_prefix}_sample_{sample}_model_fixed.cif",
            case,
            "OpenFold",
            "model_fixed_cif",
        )

        checked_copy(
            old_dir / f"{old_prefix}_sample_{sample}_confidences.json",
            pred_dir / f"{new_prefix}_sample_{sample}_confidences.json",
            case,
            "OpenFold",
            "confidence_json",
        )

        checked_copy(
            old_dir
            / f"{old_prefix}_sample_{sample}_confidences_aggregated.json",
            pred_dir
            / f"{new_prefix}_sample_{sample}_confidences_aggregated.json",
            case,
            "OpenFold",
            "aggregated_confidence_json",
        )

        # ABCFold output_models corresponds to the fixed coordinates.
        checked_copy(
            fixed_src,
            output_dir / f"openfold_model_{model_index}.cif",
            case,
            "OpenFold",
            "output_model_cif",
        )


def recover_protenix(case, condition):
    PEP = case["PEPTIDE"]
    mer = case["mer"]

    old_dir = case["protenix_source"]
    old_prefix = case["protenix_prefix"]
    seed = case["new_seed"]

    pred_dir = (
        condition
        / f"protenix_{PEP}_{mer}mer"
        / f"protenix_results_seed-{seed}"
        / f"{PEP}_{mer}mer"
        / f"seed_{seed}"
        / "predictions"
    )

    output_dir = condition / "output_models"

    for i in range(5):

        cif_src = old_dir / f"{old_prefix}_sample_{i}.cif"

        checked_copy(
            cif_src,
            pred_dir / f"{PEP}_{mer}mer_sample_{i}.cif",
            case,
            "Protenix",
            "model_cif",
        )

        checked_copy(
            cif_src,
            output_dir / f"protenix_model_{i}.cif",
            case,
            "Protenix",
            "output_model_cif",
        )

        # The old Protenix production runs contain summary confidence JSONs.
        checked_copy(
            old_dir / f"{old_prefix}_summary_confidence_sample_{i}.json",
            pred_dir / f"{PEP}_{mer}mer_summary_confidence_sample_{i}.json",
            case,
            "Protenix",
            "summary_confidence_json",
        )


def main():

    print("=" * 78)
    print("LEGACY ABCFOLD GAP-FILL")
    print("=" * 78)
    print()
    print("Recovering ONLY:")
    print("  Aβ40 5mer no templates  : Boltz + OpenFold + Protenix")
    print("  Aβ40 5mer with templates: Boltz + OpenFold + Protenix")
    print("  Aβ42 1mer with templates: Boltz + OpenFold + Protenix")
    print()
    print("NOT touching:")
    print("  Aβ42 20mer with templates")
    print("  Aβ40 50mer with templates")
    print("  Aβ42 50mer with templates")
    print()

    # -----------------------------------------------------------------
    # PRE-FLIGHT: verify every required legacy source before copying
    # anything.
    # -----------------------------------------------------------------

    missing = []

    for case in CASES:
        for d in (
            case["boltz_source"],
            case["openfold_source"],
            case["protenix_source"],
        ):
            if not d.is_dir():
                missing.append(str(d))

    if missing:
        print("ERROR: required legacy source directories are missing:")
        for p in missing:
            print(" ", p)
        sys.exit(1)

    print("PASS: all nine legacy predictor source directories found.")
    print()

    # -----------------------------------------------------------------
    # COPY
    # -----------------------------------------------------------------

    for case in CASES:

        folder_name = (
            f"{case['peptide']}_{case['mer']}mer_"
            f"{case['template']}"
        )

        condition = NEW / case["condition_dir"] / folder_name
        condition.mkdir(parents=True, exist_ok=True)

        print("-" * 78)
        print(
            f"{case['PEPTIDE']} {case['mer']}mer "
            f"{case['template']}"
        )
        print("-" * 78)

        recover_boltz(case, condition)
        print("  Boltz:    copied")

        recover_openfold(case, condition)
        print("  OpenFold: copied")

        recover_protenix(case, condition)
        print("  Protenix: copied")

        # Explicit marker within the condition.
        marker = condition / "LEGACY_GAPFILL_README.txt"

        marker.write_text(
            "LEGACY GAP-FILL CONDITION\n"
            "\n"
            "This condition was absent from the new ABCFold production run.\n"
            "The Boltz, OpenFold and Protenix models in the predictor folders\n"
            "were copied from the previous ~/abeta_project production runs.\n"
            "\n"
            "Directory and file names were standardised to match the new\n"
            "ABCFold dataset. File contents were not rewritten.\n"
            "\n"
            "See:\n"
            "  /home/benla/abeta_project_final/manifests/"
            "legacy_gapfill_manifest.tsv\n"
        )

    # -----------------------------------------------------------------
    # PROVENANCE MANIFEST
    # -----------------------------------------------------------------

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    with MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "predictor",
                "peptide",
                "template_condition",
                "oligomer_size",
                "file_type",
                "source_file",
                "destination_file",
                "provenance",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    # -----------------------------------------------------------------
    # FINAL MODEL COUNT
    # -----------------------------------------------------------------

    print()
    print("=" * 78)
    print("RECOVERED MODEL AUDIT")
    print("=" * 78)

    total_models = 0

    for case in CASES:

        folder_name = (
            f"{case['peptide']}_{case['mer']}mer_"
            f"{case['template']}"
        )

        condition = NEW / case["condition_dir"] / folder_name
        output_dir = condition / "output_models"

        counts = {}

        for predictor in ("boltz", "openfold", "protenix"):
            n = len(list(output_dir.glob(f"{predictor}_model_*.cif")))
            counts[predictor] = n
            total_models += n

        print(
            f"{case['PEPTIDE']} {case['mer']}mer "
            f"{case['template']}: "
            f"Boltz={counts['boltz']} "
            f"OpenFold={counts['openfold']} "
            f"Protenix={counts['protenix']}"
        )

        if counts != {
            "boltz": 5,
            "openfold": 5,
            "protenix": 5,
        }:
            raise RuntimeError(
                f"Unexpected model count in {condition}"
            )

    print()
    print(f"Recovered prediction models: {total_models} / 45")

    if total_models != 45:
        raise RuntimeError("Expected exactly 45 recovered predictions.")

    print()
    print("PASS: all 45 recoverable legacy models installed.")
    print()
    print(f"Provenance manifest:\n{MANIFEST}")
    print()
    print("Remaining unrecovered gaps:")
    print("  Aβ42 20mer templates — Boltz/OpenFold/Protenix")
    print("  Aβ40 50mer templates — Boltz")
    print("  Aβ42 50mer templates — Boltz")
    print("=" * 78)


if __name__ == "__main__":
    main()
