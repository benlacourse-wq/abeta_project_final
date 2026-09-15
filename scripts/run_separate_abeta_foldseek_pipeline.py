#!/usr/bin/env python3
"""
Peptide-separated Aβ Foldseek pipeline for /home/benla/abeta_project_final

PURPOSE
-------
Run the SAME representative-chain Foldseek workflow separately for exactly one
peptide (Aβ40 or Aβ42). The Foldseek settings, representative-chain rule,
best-hit ranking, QTM-cluster mapping, matrices, and figures are unchanged from
the immediately preceding combined Aβ40+Aβ42 pipeline. Only the peptide scope
is restricted so Aβ40 predictions are compared only with the 33 Aβ40
experimental representatives, or Aβ42 predictions only with the 46 Aβ42
experimental representatives.

The source project is NEVER modified. All derived files are written to:

    analysis/foldseek/{ab40_separate|ab42_separate}/
    figures/foldseek/{ab40_separate|ab42_separate}/

Canonical inputs used
---------------------
Experimental:
    experimental/structures/representatives_pdb_AB40_33/*.pdb  (Aβ40)
    experimental/structures/representatives_pdb_AB42_46/*.pdb  (Aβ42)
    experimental/foldseek/abeta_foldseek_clusters.tsv
    experimental/metadata/abeta_metadata_clean.tsv (optional enrichment)

AlphaFold3:
    AF3/{ab40,ab42}/{no_templates,with_templates}/...
    one final model CIF for each of 18 mers x 5 models x 4 conditions = 360

ABCFold:
    ABC_Fold/ab40_ab42_results/.../output_models/
    {boltz,openfold,protenix}_model_0..4.cif
    3 predictors x 18 mers x 5 models x 4 conditions = 1080

Final predicted dataset per peptide = 720 models.
Each predicted multimer contributes ONE representative peptide chain, matching
Ben's previous Aβ Foldseek workflow. Chain A is used when it is a complete
Aβ40/Aβ42 chain; otherwise the first complete 40/42-residue chain is used.
This avoids weighting larger oligomers more heavily merely because they contain
more peptide chains.

Foldseek output fields
----------------------
query,target,qtmscore,ttmscore,alntmscore,rmsd,alnlen,qcov,tcov,evalue

The Foldseek settings reproduce the prior exhaustive Aβ search logic:
    --tmscore-threshold 0
    --alignment-type 1
    -a 1
    --alignment-mode 3
    -e 10000
    -s 9.5
    -k 6
    --max-seqs 100000
    --min-ungapped-score 0
    --exhaustive-search 1

Primary experimental assignment
--------------------------------
Best experimental target is chosen by:
    1. highest query-normalised TM-score (qTM)
    2. highest query coverage
    3. longest alignment
    4. lowest RMSD

This intentionally uses qTM as the primary whole-query similarity metric.

Modes
-----
No action flag: read-only source preflight.

    python run_separate_abeta_foldseek_pipeline.py --peptide ab40
    python run_separate_abeta_foldseek_pipeline.py --peptide ab40 --prepare
    python run_separate_abeta_foldseek_pipeline.py --peptide ab40 --run-foldseek
    python run_separate_abeta_foldseek_pipeline.py --peptide ab40 --analyse
    python run_separate_abeta_foldseek_pipeline.py --peptide ab40 --all

Then repeat identically with --peptide ab42.

Use --force only to overwrite files INSIDE the dedicated derived output folders.
It never alters AF3, ABC_Fold, experimental, raw_predictions, or recovery data.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import PDB
from Bio.PDB import MMCIFParser, PDBParser, PDBIO, Select
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform


# =============================================================================
# PEPTIDE SCOPE -- the ONLY scientific scope difference from the combined run
# =============================================================================

_EARLY = argparse.ArgumentParser(add_help=False)
_EARLY.add_argument("--peptide", choices=["ab40", "ab42"], required=True)
_EARLY_ARGS, _EARLY_UNKNOWN = _EARLY.parse_known_args()
PEPTIDE = _EARLY_ARGS.peptide
PEPTIDES = [PEPTIDE]
EXPECTED_PREDICTIONS = 4 * 1 * 2 * 18 * 5  # 720
EXPECTED_EXPERIMENTAL = 33 if PEPTIDE == "ab40" else 46
EXPECTED_TOTAL = EXPECTED_PREDICTIONS + EXPECTED_EXPERIMENTAL  # 753 or 766
EXPECTED_SWITCHES = 4 * 1 * 18 * 5  # 360 paired template/no-template comparisons
PEPTIDE_LABEL = "Aβ40" if PEPTIDE == "ab40" else "Aβ42"


# =============================================================================
# FIXED PROJECT LAYOUT
# =============================================================================

PROJECT = Path("/home/benla/abeta_project_final")
ANALYSIS = PROJECT / "analysis" / "foldseek" / f"{PEPTIDE}_separate"
FIGURES = PROJECT / "figures" / "foldseek" / f"{PEPTIDE}_separate"

EXP_REPS = (
    PROJECT / "experimental" / "structures" /
    ("representatives_pdb_AB40_33" if PEPTIDE == "ab40" else "representatives_pdb_AB42_46")
)
EXP_CLUSTERS = PROJECT / "experimental" / "foldseek" / "abeta_foldseek_clusters.tsv"
EXP_METADATA = PROJECT / "experimental" / "metadata" / "abeta_metadata_clean.tsv"

REP_ROOT = ANALYSIS / "representative_chains"
REP_PREDICTED = REP_ROOT / "predicted_all"
REP_EXPERIMENTAL = REP_ROOT / "experimental"
REP_ALL = REP_ROOT / f"all_{EXPECTED_TOTAL}"
INVENTORY_TSV = ANALYSIS / f"predicted_model_inventory_{EXPECTED_PREDICTIONS}.tsv"
ALL_MANIFEST_TSV = ANALYSIS / f"all_structure_manifest_{EXPECTED_TOTAL}.tsv"

FOLDSEEK_DIR = ANALYSIS / "foldseek_results"
PRED_VS_EXP_TSV = FOLDSEEK_DIR / f"predicted_{EXPECTED_PREDICTIONS}_vs_experimental_{EXPECTED_EXPERIMENTAL}.tsv"
ALL_VS_ALL_TSV = FOLDSEEK_DIR / f"all_{EXPECTED_TOTAL}_vs_all_{EXPECTED_TOTAL}.tsv"
TMP_PRED_EXP = FOLDSEEK_DIR / "tmp_predicted_vs_experimental"
TMP_ALL = FOLDSEEK_DIR / "tmp_all_vs_all"

ASSIGN_DIR = ANALYSIS / "assignments"
ASSIGN_TSV = ASSIGN_DIR / f"model_level_assignments_{EXPECTED_PREDICTIONS}.tsv"
SWITCH_TSV = ASSIGN_DIR / "paired_template_switching.tsv"
SUMMARY_TSV = ASSIGN_DIR / "prediction_group_summary.tsv"

MATRIX_DIR = ANALYSIS / "matrices"
QTM_DIRECTED = MATRIX_DIR / f"all_{EXPECTED_TOTAL}_qtmscore_directed.csv"
QTM_SYMMETRIC = MATRIX_DIR / f"all_{EXPECTED_TOTAL}_qtmscore_symmetric.csv"
ALNTM_SYMMETRIC = MATRIX_DIR / f"all_{EXPECTED_TOTAL}_alntmscore_symmetric.csv"

MERS = list(range(1, 16)) + [20, 30, 50]
MODELS = list(range(5))
METHODS = ["AlphaFold", "Boltz", "OpenFold", "Protenix"]
METHOD_KEYS = {"AlphaFold": "af3", "Boltz": "boltz", "OpenFold": "openfold", "Protenix": "protenix"}
TEMPLATE_MODES = ["no_templates", "with_templates"]
FOLDSEEK_COLUMNS = [
    "query", "target", "qtmscore", "ttmscore", "alntmscore",
    "rmsd", "alnlen", "qcov", "tcov", "evalue",
]
NUMERIC_HIT_COLUMNS = [
    "qtmscore", "ttmscore", "alntmscore", "rmsd", "alnlen",
    "qcov", "tcov", "evalue",
]

GROUP_ORDER = [
    "AlphaFold_no_templates", "AlphaFold_with_templates",
    "Boltz_no_templates", "Boltz_with_templates",
    "OpenFold_no_templates", "OpenFold_with_templates",
    "Protenix_no_templates", "Protenix_with_templates",
]

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


# =============================================================================
# PATH / ID HELPERS
# =============================================================================

def condition_name(peptide: str, template_mode: str, mer: int) -> str:
    if template_mode == "no_templates":
        return f"{peptide}_{mer}mer_no_templates"
    return f"{peptide}_{mer}mer_templates"


def af3_path(peptide: str, template_mode: str, mer: int, model: int) -> Path:
    folder = PROJECT / "AF3" / peptide / template_mode / condition_name(peptide, template_mode, mer)
    return folder / f"fold_{condition_name(peptide, template_mode, mer)}_model_{model}.cif"


def abcfold_path(method_key: str, peptide: str, template_mode: str, mer: int, model: int) -> Path:
    pep_up = peptide.upper()
    if template_mode == "no_templates":
        group = f"ABCFold_{pep_up}_no_templates"
        cond = f"{peptide}_{mer}mer_no_templates"
    else:
        group = f"ABCFold_{pep_up}_templates"
        cond = f"{peptide}_{mer}mer_with_templates"
    return (
        PROJECT / "ABC_Fold" / "ab40_ab42_results" / group / cond /
        "output_models" / f"{method_key}_model_{model}.cif"
    )


def model_id(method: str, peptide: str, template_mode: str, mer: int, model: int) -> str:
    # Foldseek-safe, globally unique, parseable identifier.
    return f"{method}__{peptide}__{template_mode}__{mer}mer__model_{model}"


def group_key(method: str, template_mode: str) -> str:
    return f"{method}_{template_mode}"


def group_label(key: str) -> str:
    return (
        key.replace("_no_templates", "\nNo templates")
        .replace("_with_templates", "\nTemplates")
    )


def cluster_num(value: object) -> int:
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else 999999


def normalise_identifier(value: object) -> str:
    text = Path(str(value).strip()).name
    for ext in (".pdb", ".cif", ".mmcif"):
        if text.lower().endswith(ext):
            text = text[: -len(ext)]
            break
    return text.lower()


# =============================================================================
# CANONICAL SOURCE INVENTORY
# =============================================================================

def canonical_prediction_records() -> list[dict]:
    rows: list[dict] = []
    for method in METHODS:
        for peptide in PEPTIDES:
            for template_mode in TEMPLATE_MODES:
                for mer in MERS:
                    for model in MODELS:
                        if method == "AlphaFold":
                            source = af3_path(peptide, template_mode, mer, model)
                        else:
                            source = abcfold_path(METHOD_KEYS[method], peptide, template_mode, mer, model)
                        rows.append({
                            "foldseek_id": model_id(method, peptide, template_mode, mer, model),
                            "method": method,
                            "peptide": peptide,
                            "template_status": template_mode,
                            "oligomer_size": mer,
                            "model_rank": model,
                            "expected_chain_residues": 40 if peptide == "ab40" else 42,
                            "source_file": str(source),
                        })
    return rows


def source_preflight(require_foldseek: bool = False) -> pd.DataFrame:
    print("=" * 84)
    print("FINAL Aβ FOLDSEEK SOURCE PREFLIGHT")
    print("=" * 84)
    print(f"Project: {PROJECT}")

    required = [PROJECT, EXP_REPS, EXP_CLUSTERS]
    missing_roots = [p for p in required if not p.exists()]
    if missing_roots:
        raise FileNotFoundError("Missing required path(s):\n  " + "\n  ".join(map(str, missing_roots)))

    rows = canonical_prediction_records()
    df = pd.DataFrame(rows)
    missing = [Path(p) for p in df["source_file"] if not Path(p).is_file()]
    exp_files = sorted(EXP_REPS.glob("*.pdb"))

    print(f"Predicted structures expected: {EXPECTED_PREDICTIONS}")
    print(f"Predicted structures found:    {len(df) - len(missing)}")
    print(f"Experimental reps expected:    {EXPECTED_EXPERIMENTAL}")
    print(f"Experimental reps found:       {len(exp_files)}")
    print(f"Total canonical structures:    {len(df) - len(missing) + len(exp_files)}")

    if len(df) != EXPECTED_PREDICTIONS:
        raise RuntimeError(f"Internal inventory error: expected {EXPECTED_PREDICTIONS}, built {len(df)}")
    if missing:
        shown = "\n  ".join(str(p) for p in missing[:30])
        raise FileNotFoundError(f"Missing {len(missing)} canonical predicted structure(s):\n  {shown}")
    if len(exp_files) != EXPECTED_EXPERIMENTAL:
        raise RuntimeError(f"Expected {EXPECTED_EXPERIMENTAL} experimental representative PDBs, found {len(exp_files)}")

    counts = df.groupby(["method", "peptide", "template_status"]).size()
    bad = counts[counts != 90]
    print("\nCounts per predictor / peptide / template condition:")
    print(counts.to_string())
    if not bad.empty:
        raise RuntimeError("Every predictor/peptide/template condition must contain 90 models.")

    foldseek = shutil.which("foldseek")
    print(f"\nFoldseek executable: {foldseek or 'NOT FOUND'}")
    if foldseek:
        proc = subprocess.run([foldseek, "version"], text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=False)
        print(f"Foldseek version:    {proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else 'unknown'}")
    elif require_foldseek:
        raise RuntimeError(
            "Foldseek is not on PATH. Create/activate a Foldseek environment before --run-foldseek."
        )

    print(f"\nPREFLIGHT PASSED: {PEPTIDE_LABEL} canonical source dataset is {EXPECTED_PREDICTIONS} predictions + {EXPECTED_EXPERIMENTAL} experimental reps = {EXPECTED_TOTAL}.")
    print("Raw/back-up/template/recovery copies are deliberately excluded.\n")
    return df


# =============================================================================
# REPRESENTATIVE-CHAIN EXTRACTION
# =============================================================================

class ChainSelect(Select):
    def __init__(self, chain_id: str):
        self.chain_id = chain_id
    def accept_chain(self, chain):
        return 1 if chain.id == self.chain_id else 0


def residue_count(chain) -> int:
    return sum(1 for residue in chain if residue.id[0] == " " and PDB.is_aa(residue))


STANDARD_AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def _as_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _cif_column(cif: dict, *keys: str, n: int | None = None, default: str = "") -> list[str]:
    """Return the first available mmCIF column, padded to n values if requested."""
    for key in keys:
        if key in cif:
            values = [str(x) for x in _as_list(cif[key])]
            if n is not None:
                if len(values) == 1 and n > 1:
                    values = values * n
                elif len(values) < n:
                    values = values + [default] * (n - len(values))
            return values
    return [default] * n if n is not None else []


def _float_or(value: str, default: float) -> float:
    try:
        if value in {"", ".", "?"}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_mmcif_without_biopython_structure_parser(
    source: Path,
    output: Path,
    expected_len: int,
) -> tuple[str, int, str]:
    """
    Minimal, robust mmCIF -> single-chain PDB conversion.

    Some predictor mmCIF files omit optional atom_site columns such as
    _atom_site.occupancy. Bio.PDB.MMCIFParser treats that column as required and
    raises KeyError even though the coordinates themselves are perfectly usable.
    MMCIF2Dict is more permissive, so this fallback reads atom_site directly,
    supplies occupancy=1.00 only in the DERIVED PDB, and never changes the source CIF.
    """
    cif = MMCIF2Dict(str(source))

    xs = _cif_column(cif, "_atom_site.Cartn_x")
    if not xs:
        raise ValueError("mmCIF has no _atom_site.Cartn_x coordinate column")
    n = len(xs)

    ys = _cif_column(cif, "_atom_site.Cartn_y", n=n)
    zs = _cif_column(cif, "_atom_site.Cartn_z", n=n)
    groups = _cif_column(cif, "_atom_site.group_PDB", n=n, default="ATOM")
    atom_names = _cif_column(cif, "_atom_site.auth_atom_id", "_atom_site.label_atom_id", n=n)
    resnames = _cif_column(cif, "_atom_site.auth_comp_id", "_atom_site.label_comp_id", n=n)
    auth_chains = _cif_column(cif, "_atom_site.auth_asym_id", n=n)
    label_chains = _cif_column(cif, "_atom_site.label_asym_id", n=n)
    auth_seq = _cif_column(cif, "_atom_site.auth_seq_id", n=n)
    label_seq = _cif_column(cif, "_atom_site.label_seq_id", n=n)
    inscodes = _cif_column(cif, "_atom_site.pdbx_PDB_ins_code", n=n, default="")
    altlocs = _cif_column(cif, "_atom_site.label_alt_id", n=n, default="")
    elements = _cif_column(cif, "_atom_site.type_symbol", n=n, default="")
    occupancies = _cif_column(cif, "_atom_site.occupancy", n=n, default="1.0")
    bfactors = _cif_column(cif, "_atom_site.B_iso_or_equiv", n=n, default="0.0")
    models = _cif_column(cif, "_atom_site.pdbx_PDB_model_num", n=n, default="1")

    atom_rows = []
    residue_order_by_chain: dict[str, list[tuple[str, str, str]]] = {}
    residue_seen_by_chain: dict[str, set[tuple[str, str, str]]] = {}

    for i in range(n):
        group = groups[i].upper()
        if group not in {"ATOM", "HETATM"}:
            continue
        if str(models[i]) not in {"1", "1.0", ".", "?", ""}:
            continue

        resname = resnames[i].strip().upper()
        if resname not in STANDARD_AA3:
            continue

        chain = auth_chains[i].strip()
        if chain in {"", ".", "?"}:
            chain = label_chains[i].strip()
        if chain in {"", ".", "?"}:
            chain = "A"

        seq = auth_seq[i].strip()
        if seq in {"", ".", "?"}:
            seq = label_seq[i].strip()
        ins = inscodes[i].strip()
        if ins in {".", "?"}:
            ins = ""

        residue_key = (seq, ins, resname)
        residue_seen_by_chain.setdefault(chain, set())
        residue_order_by_chain.setdefault(chain, [])
        if residue_key not in residue_seen_by_chain[chain]:
            residue_seen_by_chain[chain].add(residue_key)
            residue_order_by_chain[chain].append(residue_key)

        alt = altlocs[i].strip()
        if alt not in {"", ".", "?", "A", "1"}:
            continue

        atom_rows.append({
            "chain": chain,
            "residue_key": residue_key,
            "atom_name": atom_names[i].strip(),
            "resname": resname,
            "x": _float_or(xs[i], 0.0),
            "y": _float_or(ys[i], 0.0),
            "z": _float_or(zs[i], 0.0),
            "occupancy": _float_or(occupancies[i], 1.0),
            "bfactor": _float_or(bfactors[i], 0.0),
            "element": elements[i].strip().upper(),
        })

    if not residue_order_by_chain:
        raise ValueError("no standard amino-acid chains found in mmCIF atom_site")

    lengths = {chain: len(residues) for chain, residues in residue_order_by_chain.items()}
    eligible = [chain for chain, length in lengths.items() if length == expected_len]
    if not eligible:
        raise ValueError(
            f"no complete {expected_len}-residue peptide chain in mmCIF fallback; "
            f"chain lengths={lengths}"
        )

    chosen = "A" if "A" in eligible else eligible[0]
    residue_numbers = {
        key: idx for idx, key in enumerate(residue_order_by_chain[chosen], start=1)
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    serial = 1
    with output.open("w", encoding="ascii", errors="replace") as handle:
        handle.write(f"REMARK derived from {source.name}; selected source chain {chosen}\n")
        handle.write("REMARK missing mmCIF occupancy values, if any, were written as 1.00 in this derived PDB\n")
        for row in atom_rows:
            if row["chain"] != chosen:
                continue
            atom_name = row["atom_name"] or row["element"] or "X"
            resseq = residue_numbers[row["residue_key"]]
            element = row["element"] or re.sub(r"[^A-Za-z]", "", atom_name)[:1].upper()
            # Representative PDBs use chain A consistently; filename preserves model identity.
            handle.write(
                f"ATOM  {serial:5d} {atom_name:>4s} {row['resname']:>3s} A{resseq:4d}    "
                f"{row['x']:8.3f}{row['y']:8.3f}{row['z']:8.3f}"
                f"{row['occupancy']:6.2f}{row['bfactor']:6.2f}          {element:>2s}\n"
            )
            serial += 1
        handle.write("TER\nEND\n")

    if serial == 1:
        raise ValueError(f"selected chain {chosen} produced no atoms")

    return chosen, len(lengths), ";".join(f"{k}:{v}" for k, v in lengths.items())


def extract_one_chain(source: Path, output: Path, expected_len: int) -> tuple[str, int, str]:
    """
    Extract one complete Aβ chain.

    First try Bio.PDB's normal structure parsers. If an mmCIF is missing an
    optional atom_site field that Bio.PDB assumes exists (notably occupancy),
    fall back to a direct MMCIF2Dict reader and write a derived PDB without
    modifying the source CIF.
    """
    if source.suffix.lower() not in {".cif", ".mmcif"}:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(source.stem, str(source))
        model = next(structure.get_models())
        chains = list(model.get_chains())
        if not chains:
            raise ValueError("no chains found")
        lengths = {c.id: residue_count(c) for c in chains}
        eligible = [c for c in chains if lengths[c.id] == expected_len]
        if not eligible:
            raise ValueError(f"no complete {expected_len}-residue peptide chain; chain lengths={lengths}")
        chosen = next((c for c in eligible if c.id == "A"), eligible[0])
        output.parent.mkdir(parents=True, exist_ok=True)
        io = PDBIO()
        io.set_structure(model)
        io.save(str(output), ChainSelect(chosen.id))
        return chosen.id, len(chains), ";".join(f"{k}:{v}" for k, v in lengths.items())

    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(source.stem, str(source))
        model = next(structure.get_models())
        chains = list(model.get_chains())
        if not chains:
            raise ValueError("no chains found")
        lengths = {c.id: residue_count(c) for c in chains}
        eligible = [c for c in chains if lengths[c.id] == expected_len]
        if not eligible:
            raise ValueError(f"no complete {expected_len}-residue peptide chain; chain lengths={lengths}")
        chosen = next((c for c in eligible if c.id == "A"), eligible[0])
        output.parent.mkdir(parents=True, exist_ok=True)
        io = PDBIO()
        io.set_structure(model)
        io.save(str(output), ChainSelect(chosen.id))
        return chosen.id, len(chains), ";".join(f"{k}:{v}" for k, v in lengths.items())
    except (KeyError, IndexError, TypeError, ValueError) as normal_error:
        try:
            return _extract_mmcif_without_biopython_structure_parser(
                source, output, expected_len
            )
        except Exception as fallback_error:
            raise RuntimeError(
                f"normal mmCIF parser failed ({type(normal_error).__name__}: {normal_error}); "
                f"fallback parser also failed ({type(fallback_error).__name__}: {fallback_error})"
            ) from fallback_error


def safe_copy(src: Path, dst: Path, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return
    shutil.copy2(src, dst)


def prepare_representatives(force: bool = False) -> None:
    source_df = source_preflight(require_foldseek=False)
    print("=" * 84)
    print("PREPARING REPRESENTATIVE-CHAIN DATASET")
    print("=" * 84)

    for folder in [REP_PREDICTED, REP_EXPERIMENTAL, REP_ALL]:
        folder.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, rec in enumerate(source_df.to_dict(orient="records"), 1):
        src = Path(rec["source_file"])
        out = REP_PREDICTED / f"{rec['foldseek_id']}.pdb"
        if out.exists() and not force:
            # Re-parse existing output to validate it remains a single complete peptide chain.
            parser = PDBParser(QUIET=True)
            model = next(parser.get_structure(out.stem, str(out)).get_models())
            chains = list(model.get_chains())
            if len(chains) != 1 or residue_count(chains[0]) != rec["expected_chain_residues"]:
                raise RuntimeError(f"Existing representative is invalid; rerun with --force: {out}")
            chosen, n_chains, chain_lengths = chains[0].id, "existing", f"{chains[0].id}:{residue_count(chains[0])}"
        else:
            try:
                chosen, n_chains, chain_lengths = extract_one_chain(
                    src, out, int(rec["expected_chain_residues"])
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Representative extraction failed at model {i}/{len(source_df)}\n"
                    f"foldseek_id: {rec['foldseek_id']}\n"
                    f"source: {src}\n"
                    f"reason: {type(exc).__name__}: {exc}"
                ) from exc

        safe_copy(out, REP_ALL / out.name, force)
        rec.update({
            "chosen_chain": chosen,
            "n_chains_in_model": n_chains,
            "source_chain_lengths": chain_lengths,
            "representative_pdb": str(out),
            "status": "ok",
        })
        rows.append(rec)
        if i % 100 == 0 or i == len(source_df):
            print(f"  predicted representatives: {i}/{len(source_df)}")

    inv = pd.DataFrame(rows).sort_values(
        ["method", "peptide", "template_status", "oligomer_size", "model_rank"]
    )
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    inv.to_csv(INVENTORY_TSV, sep="\t", index=False)

    exp_rows = []
    for src in sorted(EXP_REPS.glob("*.pdb")):
        safe_copy(src, REP_EXPERIMENTAL / src.name, force)
        safe_copy(src, REP_ALL / src.name, force)
        exp_rows.append({
            "foldseek_id": src.stem,
            "source_type": "Experimental",
            "source_file": str(src),
            "representative_pdb": str(REP_EXPERIMENTAL / src.name),
        })

    all_rows = [
        {
            "foldseek_id": r["foldseek_id"], "source_type": r["method"],
            "source_file": r["source_file"], "representative_pdb": r["representative_pdb"],
        }
        for r in rows
    ] + exp_rows
    all_manifest = pd.DataFrame(all_rows)
    all_manifest.to_csv(ALL_MANIFEST_TSV, sep="\t", index=False)

    pred_count = len(list(REP_PREDICTED.glob("*.pdb")))
    exp_count = len(list(REP_EXPERIMENTAL.glob("*.pdb")))
    all_count = len(list(REP_ALL.glob("*.pdb")))
    print(f"\nPredicted representative PDBs:    {pred_count}")
    print(f"Experimental representative PDBs: {exp_count}")
    print(f"All representative PDBs:          {all_count}")
    print(f"Inventory: {INVENTORY_TSV}")
    if (pred_count, exp_count, all_count) != (EXPECTED_PREDICTIONS, EXPECTED_EXPERIMENTAL, EXPECTED_TOTAL):
        raise RuntimeError("Prepared representative dataset has incorrect counts.")
    print("REPRESENTATIVE DATASET COMPLETE. Original structures were not modified.\n")


# =============================================================================
# FOLDSEEK
# =============================================================================

def foldseek_command(query: Path, target: Path, output: Path, tmp: Path) -> list[str]:
    exe = shutil.which("foldseek")
    if not exe:
        raise RuntimeError("Foldseek is not on PATH.")
    return [
        exe, "easy-search", str(query), str(target), str(output), str(tmp),
        "--tmscore-threshold", "0",
        "--alignment-type", "1",
        "-a", "1",
        "--alignment-mode", "3",
        "-e", "10000",
        "-s", "9.5",
        "-k", "6",
        "--max-seqs", "100000",
        "--min-ungapped-score", "0",
        "--exhaustive-search", "1",
        "--format-output", ",".join(FOLDSEEK_COLUMNS),
    ]


def run_one_foldseek(query: Path, target: Path, output: Path, tmp: Path, force: bool) -> None:
    if output.exists() and output.stat().st_size > 0 and not force:
        print(f"Reusing existing Foldseek output: {output}")
        return
    if force and output.exists():
        output.unlink()
    if force and tmp.exists():
        shutil.rmtree(tmp)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = foldseek_command(query, target, output, tmp)
    print("\nRunning:")
    print(" ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Foldseek did not create a valid result: {output}")


def run_foldseek(force: bool = False) -> None:
    source_preflight(require_foldseek=True)
    if not INVENTORY_TSV.is_file() or len(list(REP_PREDICTED.glob("*.pdb"))) != EXPECTED_PREDICTIONS:
        raise RuntimeError("Representative dataset is not prepared. Run --prepare first.")
    if len(list(REP_EXPERIMENTAL.glob("*.pdb"))) != EXPECTED_EXPERIMENTAL or len(list(REP_ALL.glob("*.pdb"))) != EXPECTED_TOTAL:
        raise RuntimeError("Prepared representative folders have incorrect counts. Run --prepare.")

    print("=" * 84)
    print(f"FOLDSEEK 1/2 — {PEPTIDE_LABEL}: {EXPECTED_PREDICTIONS} PREDICTIONS vs {EXPECTED_EXPERIMENTAL} EXPERIMENTAL REPRESENTATIVES")
    print("=" * 84)
    run_one_foldseek(REP_PREDICTED, REP_EXPERIMENTAL, PRED_VS_EXP_TSV, TMP_PRED_EXP, force)

    print("\n" + "=" * 84)
    print(f"FOLDSEEK 2/2 — {PEPTIDE_LABEL}: ALL {EXPECTED_TOTAL} REPRESENTATIVE STRUCTURES vs ALL {EXPECTED_TOTAL}")
    print("=" * 84)
    run_one_foldseek(REP_ALL, REP_ALL, ALL_VS_ALL_TSV, TMP_ALL, force)
    print("\nFOLDSEEK RUNS COMPLETE.")
    print(PRED_VS_EXP_TSV)
    print(ALL_VS_ALL_TSV)


# =============================================================================
# FOLDSEEK PARSING / CLUSTER MAPPING
# =============================================================================

def read_foldseek(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t", names=FOLDSEEK_COLUMNS, header=None)
    # Be tolerant if user/runtime produced a headered table.
    if len(df) and str(df.iloc[0]["query"]).lower() == "query":
        df = df.iloc[1:].copy()
    for col in NUMERIC_HIT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["qtmscore", "ttmscore", "alntmscore"]:
        df[col] = df[col].clip(lower=0, upper=1)
    df["query"] = df["query"].astype(str).map(normalise_identifier)
    df["target"] = df["target"].astype(str).map(normalise_identifier)
    return df


def load_cluster_map() -> tuple[dict[str, str], pd.DataFrame]:
    c = pd.read_csv(EXP_CLUSTERS, sep="\t")
    lower = {x.lower(): x for x in c.columns}
    structure_col = next((lower[k] for k in ["structure", "target", "representative", "id"] if k in lower), None)
    cluster_col = next((lower[k] for k in ["cluster", "qtm_cluster", "structural_cluster"] if k in lower), None)
    if not structure_col or not cluster_col:
        raise ValueError(
            f"Could not identify structure/cluster columns in {EXP_CLUSTERS}. Columns={list(c.columns)}"
        )
    c = c[[structure_col, cluster_col]].copy()
    c["norm"] = c[structure_col].map(normalise_identifier)
    if c["norm"].duplicated().any():
        raise ValueError("Experimental cluster table contains duplicate structure identifiers.")
    mapping = dict(zip(c["norm"], c[cluster_col].astype(str)))

    # PDB-id fallback is allowed only where unambiguous.
    pdb_groups = {}
    for norm, cluster in mapping.items():
        pdb_id = re.split(r"_chain_|_", norm, maxsplit=1)[0]
        pdb_groups.setdefault(pdb_id, set()).add(cluster)
    for pdb_id, clusters in pdb_groups.items():
        if len(clusters) == 1:
            mapping.setdefault(pdb_id, next(iter(clusters)))
    return mapping, c


def build_assignments() -> pd.DataFrame:
    inv = pd.read_csv(INVENTORY_TSV, sep="\t")
    inv["query_norm"] = inv["foldseek_id"].map(normalise_identifier)
    hits = read_foldseek(PRED_VS_EXP_TSV)

    expected_pairs = EXPECTED_PREDICTIONS * EXPECTED_EXPERIMENTAL
    n_pairs = len(hits.drop_duplicates(["query", "target"]))
    print(f"Prediction-vs-experimental Foldseek rows: {len(hits)}")
    print(f"Unique query-target pairs:               {n_pairs}/{expected_pairs}")

    # qTM primary; qcov, alnlen, low RMSD are deterministic tie-breaks.
    best = (
        hits.sort_values(
            ["query", "qtmscore", "qcov", "alnlen", "rmsd"],
            ascending=[True, False, False, False, True],
            na_position="last",
        )
        .drop_duplicates("query", keep="first")
        .copy()
    )

    cluster_map, _ = load_cluster_map()
    best["qtm_cluster"] = best["target"].map(cluster_map)
    missing_cluster = best[best["qtm_cluster"].isna()]["target"].drop_duplicates().tolist()
    if missing_cluster:
        # Try PDB-only fallback.
        best.loc[best["qtm_cluster"].isna(), "qtm_cluster"] = (
            best.loc[best["qtm_cluster"].isna(), "target"]
            .map(lambda x: cluster_map.get(re.split(r"_chain_|_", x, maxsplit=1)[0]))
        )
    still_missing = best[best["qtm_cluster"].isna()]["target"].drop_duplicates().tolist()
    if still_missing:
        raise ValueError(
            "Best-hit experimental targets could not be mapped to existing PDB-derived clusters:\n  "
            + "\n  ".join(still_missing[:30])
        )

    assignments = inv.merge(best, left_on="query_norm", right_on="query", how="left", validate="one_to_one")
    if assignments["qtmscore"].isna().any():
        missing = assignments.loc[assignments["qtmscore"].isna(), "foldseek_id"].tolist()
        raise RuntimeError(f"{len(missing)} predicted models have no Foldseek experimental hit.")

    assignments["method"] = assignments["method"].astype(str)
    assignments["group"] = assignments.apply(lambda r: group_key(r["method"], r["template_status"]), axis=1)
    assignments["target_clean"] = assignments["target"]
    assignments["best_pdb_id"] = assignments["target"].map(lambda x: re.split(r"_chain_|_", x, maxsplit=1)[0].upper())

    # Optional experimental metadata enrichment.
    if EXP_METADATA.is_file():
        try:
            meta = pd.read_csv(EXP_METADATA, sep="\t")
            if "structure" in meta.columns:
                meta = meta.copy()
                meta["target_norm"] = meta["structure"].map(normalise_identifier)
                meta = meta.drop_duplicates("target_norm")
                useful = [c for c in meta.columns if c != "structure"]
                assignments = assignments.merge(
                    meta[useful], left_on="target", right_on="target_norm", how="left", suffixes=("", "_experimental")
                )
        except Exception as exc:
            print(f"WARNING: experimental metadata enrichment skipped: {exc}")

    ASSIGN_DIR.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(ASSIGN_TSV, sep="\t", index=False)
    return assignments


def build_template_switching(assignments: pd.DataFrame) -> pd.DataFrame:
    key = ["method", "peptide", "oligomer_size", "model_rank"]
    nt = assignments[assignments["template_status"] == "no_templates"].copy()
    wt = assignments[assignments["template_status"] == "with_templates"].copy()
    keep = key + ["qtm_cluster", "qtmscore", "target_clean", "rmsd", "qcov", "alnlen"]
    paired = nt[keep].merge(wt[keep], on=key, suffixes=("_no_templates", "_templates"), validate="one_to_one")
    paired["cluster_switched"] = paired["qtm_cluster_no_templates"] != paired["qtm_cluster_templates"]
    paired["delta_qtmscore"] = paired["qtmscore_templates"] - paired["qtmscore_no_templates"]
    paired.to_csv(SWITCH_TSV, sep="\t", index=False)
    return paired


# =============================================================================
# ALL-vs-ALL MATRICES
# =============================================================================

def build_symmetric_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(ALL_MANIFEST_TSV, sep="\t")
    ids = [normalise_identifier(x) for x in manifest["foldseek_id"]]
    if len(ids) != EXPECTED_TOTAL or len(set(ids)) != EXPECTED_TOTAL:
        raise RuntimeError(f"All-structure manifest does not contain {EXPECTED_TOTAL} unique Foldseek IDs.")

    hits = read_foldseek(ALL_VS_ALL_TSV)
    hits = hits.drop_duplicates(["query", "target"], keep="first")
    nonself = hits[hits["query"] != hits["target"]]
    expected_nonself = EXPECTED_TOTAL * (EXPECTED_TOTAL - 1)
    print(f"All-vs-all unique non-self directed pairs: {len(nonself)}/{expected_nonself}")

    def matrix(score: str) -> pd.DataFrame:
        pivot = hits.pivot(index="query", columns="target", values=score)
        pivot = pivot.reindex(index=ids, columns=ids)

        # Pandas 2.x/3.x can expose DataFrame backing arrays as read-only
        # (especially with Copy-on-Write enabled).  Never mutate .values
        # directly; create an explicitly writable NumPy copy instead.
        arr = pivot.fillna(0.0).to_numpy(dtype=float, copy=True)
        np.fill_diagonal(arr, 1.0)
        return pd.DataFrame(arr, index=ids, columns=ids)

    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    qdir = matrix("qtmscore")
    adir = matrix("alntmscore")

    qarr = (
        qdir.to_numpy(dtype=float, copy=True)
        + qdir.T.to_numpy(dtype=float, copy=True)
    ) / 2.0
    aarr = (
        adir.to_numpy(dtype=float, copy=True)
        + adir.T.to_numpy(dtype=float, copy=True)
    ) / 2.0
    np.fill_diagonal(qarr, 1.0)
    np.fill_diagonal(aarr, 1.0)
    qsym = pd.DataFrame(qarr, index=ids, columns=ids)
    asym = pd.DataFrame(aarr, index=ids, columns=ids)
    qdir.to_csv(QTM_DIRECTED, float_format="%.6f")
    qsym.to_csv(QTM_SYMMETRIC, float_format="%.6f")
    asym.to_csv(ALNTM_SYMMETRIC, float_format="%.6f")

    coverage = pd.DataFrame([{
        "n_structures": EXPECTED_TOTAL,
        "expected_nonself_directed_pairs": expected_nonself,
        "observed_nonself_directed_pairs": len(nonself),
        "coverage_percent": 100.0 * len(nonself) / expected_nonself,
        "missing_pairs_filled_with_zero_in_matrices": expected_nonself - len(nonself),
    }])
    coverage.to_csv(MATRIX_DIR / "all_vs_all_pair_coverage.tsv", sep="\t", index=False)
    return qsym, asym


# =============================================================================
# FIGURES
# =============================================================================

def savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def add_heatmap_text(ax, arr, fmt="{:.0f}", threshold=0.0) -> None:
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isnan(v) or abs(v) < threshold:
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7, color="black")


def assignment_tables(df: pd.DataFrame):
    groups = [g for g in GROUP_ORDER if g in set(df["group"])]
    clusters = sorted(df["qtm_cluster"].dropna().unique(), key=cluster_num)
    counts = df.groupby(["group", "qtm_cluster"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=groups, columns=clusters, fill_value=0)
    perc = counts.div(counts.sum(axis=1), axis=0) * 100
    return groups, clusters, counts, perc


def figure_1_counts(df: pd.DataFrame, out: Path) -> None:
    groups, clusters, counts, _ = assignment_tables(df)
    width = max(14, 0.38 * len(clusters) + 7)
    fig, ax = plt.subplots(figsize=(width, 5.4))
    im = ax.imshow(counts.values, cmap="viridis", aspect="auto", vmin=0)
    ax.set_title("Predicted model occupancy across PDB-derived QTM structural clusters")
    ax.set_xticks(range(len(clusters))); ax.set_xticklabels(clusters, rotation=45, ha="right")
    ax.set_yticks(range(len(groups))); ax.set_yticklabels([group_label(g) for g in groups])
    ax.set_xlabel("PDB-derived QTM cluster"); ax.set_ylabel("Prediction method")
    add_heatmap_text(ax, counts.values, "{:.0f}", 1)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("Number of models")
    savefig(fig, out)


def figure_2_percent(df: pd.DataFrame, out: Path) -> None:
    groups, clusters, _, perc = assignment_tables(df)
    width = max(14, 0.38 * len(clusters) + 7)
    fig, ax = plt.subplots(figsize=(width, 5.4))
    im = ax.imshow(perc.values, cmap="magma", aspect="auto", vmin=0)
    ax.set_title("Relative occupancy of PDB-derived QTM structural clusters")
    ax.set_xticks(range(len(clusters))); ax.set_xticklabels(clusters, rotation=45, ha="right")
    ax.set_yticks(range(len(groups))); ax.set_yticklabels([group_label(g) for g in groups])
    ax.set_xlabel("PDB-derived QTM cluster"); ax.set_ylabel("Prediction method")
    add_heatmap_text(ax, perc.values, "{:.0f}", 1)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("Models assigned to cluster (%)")
    savefig(fig, out)


def figure_3_delta(df: pd.DataFrame, out: Path) -> None:
    _, clusters, _, perc = assignment_tables(df)
    rows, labels = [], []
    for method in METHODS:
        temp, nt = f"{method}_with_templates", f"{method}_no_templates"
        if temp in perc.index and nt in perc.index:
            rows.append((perc.loc[temp] - perc.loc[nt]).values)
            labels.append(method)
    delta = np.vstack(rows)
    vmax = max(1.0, float(np.nanmax(np.abs(delta))))
    width = max(14, 0.38 * len(clusters) + 7)
    fig, ax = plt.subplots(figsize=(width, 4.2))
    im = ax.imshow(delta, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title("Template-associated change in QTM cluster occupancy")
    ax.set_xticks(range(len(clusters))); ax.set_xticklabels(clusters, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("PDB-derived QTM cluster"); ax.set_ylabel("Prediction method")
    add_heatmap_text(ax, delta, "{:+.0f}", 2)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("Templates − no templates (% points)")
    savefig(fig, out)


def figure_4_stacked(df: pd.DataFrame, out: Path) -> None:
    groups, clusters, _, perc = assignment_tables(df)
    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(len(groups))
    cmap = plt.get_cmap("tab20", max(1, len(clusters)))
    for i, cluster in enumerate(clusters):
        vals = perc[cluster].values
        ax.bar(range(len(groups)), vals, bottom=bottom, label=cluster, color=cmap(i))
        bottom += vals
    ax.set_ylim(0, 100); ax.set_ylabel("Models assigned to cluster (%)")
    ax.set_title("Structural-cluster composition of each prediction group")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([group_label(g) for g in groups], rotation=30, ha="right")
    ax.legend(title="QTM cluster", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=max(1, math.ceil(len(clusters)/18)))
    savefig(fig, out)


def figure_5_oligomer(df: pd.DataFrame, out: Path) -> None:
    clusters = sorted(df["qtm_cluster"].unique(), key=cluster_num)
    tab = df.groupby(["oligomer_size", "qtm_cluster"]).size().unstack(fill_value=0)
    tab = tab.reindex(index=MERS, columns=clusters, fill_value=0)
    pct = tab.div(tab.sum(axis=1), axis=0) * 100
    width = max(14, 0.38 * len(clusters) + 7)
    fig, ax = plt.subplots(figsize=(width, 8))
    im = ax.imshow(pct.values, cmap="viridis", aspect="auto", vmin=0, vmax=max(1, np.nanmax(pct.values)))
    ax.set_title("Oligomer size dependence of QTM structural-cluster assignment")
    ax.set_xticks(range(len(clusters))); ax.set_xticklabels(clusters, rotation=45, ha="right")
    ax.set_yticks(range(len(MERS))); ax.set_yticklabels([f"{x}mer" for x in MERS])
    ax.set_xlabel("PDB-derived QTM cluster"); ax.set_ylabel("Oligomer size")
    add_heatmap_text(ax, pct.values, "{:.0f}", 10)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("Models assigned to cluster (%)")
    savefig(fig, out)


def figure_6_dendrogram(df: pd.DataFrame, out: Path) -> None:
    groups, _, _, perc = assignment_tables(df)
    Z = linkage(perc.values, method="average", metric="euclidean")
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, labels=[group_label(g).replace("\n", " ") for g in groups], leaf_rotation=35, ax=ax)
    ax.set_title("Similarity of prediction methods based on QTM cluster occupancy")
    ax.set_ylabel("Euclidean distance between occupancy profiles")
    savefig(fig, out)


def figure_7_boxplot(df: pd.DataFrame, out: Path) -> None:
    groups = [g for g in GROUP_ORDER if g in set(df["group"])]
    data = [df.loc[df["group"] == g, "qtmscore"].values for g in groups]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(data, showfliers=False)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1.2)
    ax.text(0.5, 0.505, "qTM = 0.5", color="grey", va="bottom")
    ax.set_title("Quality of best PDB representative match")
    ax.set_ylabel("Best query-normalised TM-score (qtmscore)")
    ax.set_xticks(range(1, len(groups) + 1)); ax.set_xticklabels([group_label(g) for g in groups], rotation=35, ha="right")
    savefig(fig, out)


def figure_8_by_mer(df: pd.DataFrame, out: Path) -> None:
    groups = [g for g in GROUP_ORDER if g in set(df["group"])]
    fig, ax = plt.subplots(figsize=(12, 6))
    for g in groups:
        s = df[df["group"] == g].groupby("oligomer_size")["qtmscore"].mean().reindex(MERS)
        ax.plot(MERS, s.values, marker="o", label=group_label(g).replace("\n", " "))
    ax.set_title("Best-match QTM-score across oligomer sizes")
    ax.set_xlabel("Oligomer size"); ax.set_ylabel("Mean best qtmscore")
    ax.set_xticks(MERS)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(fig, out)


def figure_9_targets(df: pd.DataFrame, out: Path) -> None:
    top = (
        df.groupby(["target_clean", "qtm_cluster"]).size().reset_index(name="n")
        .sort_values("n", ascending=False).head(25).sort_values("n")
    )
    labels = top["target_clean"] + " (" + top["qtm_cluster"] + ")"
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh(range(len(top)), top["n"])
    ax.set_yticks(range(len(top))); ax.set_yticklabels(labels)
    ax.set_xlabel("Number of predicted models assigned")
    ax.set_title("Most frequently selected experimental PDB representatives")
    savefig(fig, out)


def figure_10_global_all_vs_all(qsym: pd.DataFrame, out: Path) -> None:
    # Work on a writable NumPy copy; do not mutate a pandas backing view.
    sim = qsym.to_numpy(dtype=float, copy=True)
    np.clip(sim, 0.0, 1.0, out=sim)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    Z = linkage(squareform(dist, checks=False), method="average")
    order = leaves_list(Z)
    ordered = qsym.values[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(ordered, cmap="viridis", vmin=0, vmax=1, aspect="auto", interpolation="nearest", rasterized=True)
    ax.set_title(f"{PEPTIDE_LABEL} all-vs-all representative-chain structural similarity\n(symmetrised query-normalised TM-score; {EXPECTED_TOTAL} structures)")
    ax.set_xlabel("Hierarchically ordered structures"); ax.set_ylabel("Hierarchically ordered structures")
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("Symmetrised qTM")
    savefig(fig, out)
    # Save tree order/linkage for reproducibility.
    pd.DataFrame({"order": np.arange(len(order)), "matrix_index": order, "foldseek_id": qsym.index[order]}).to_csv(
        MATRIX_DIR / f"all_{EXPECTED_TOTAL}_hierarchical_order.tsv", sep="\t", index=False
    )
    pd.DataFrame(Z, columns=["left", "right", "distance", "n_members"]).to_csv(
        MATRIX_DIR / f"all_{EXPECTED_TOTAL}_linkage_average_sym_qtm.tsv", sep="\t", index=False
    )


def make_summary(assignments: pd.DataFrame, switches: pd.DataFrame) -> None:
    rows = []
    for g in GROUP_ORDER:
        sub = assignments[assignments["group"] == g]
        if sub.empty:
            continue
        counts = sub["qtm_cluster"].value_counts()
        rows.append({
            "group": g,
            "n_models": len(sub),
            "median_best_qtm": sub["qtmscore"].median(),
            "mean_best_qtm": sub["qtmscore"].mean(),
            "mean_qcov": sub["qcov"].mean(),
            "mean_rmsd": sub["rmsd"].mean(),
            "n_qtm_clusters": sub["qtm_cluster"].nunique(),
            "dominant_cluster": counts.index[0],
            "dominant_cluster_n": int(counts.iloc[0]),
            "dominant_cluster_percent": 100.0 * counts.iloc[0] / len(sub),
        })
    pd.DataFrame(rows).to_csv(SUMMARY_TSV, sep="\t", index=False)

    switch_summary = (
        switches.groupby("method")["cluster_switched"]
        .agg(["count", "sum", "mean"]).reset_index()
        .rename(columns={"count": "paired_models", "sum": "switches", "mean": "switch_fraction"})
    )
    switch_summary["switch_percent"] = switch_summary["switch_fraction"] * 100
    switch_summary.to_csv(ASSIGN_DIR / "template_switch_summary_by_method.tsv", sep="\t", index=False)


def create_figures(assignments: pd.DataFrame, qsym: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_1_counts(assignments, FIGURES / "figure_1_cluster_occupancy_counts.png")
    figure_2_percent(assignments, FIGURES / "figure_2_cluster_occupancy_percentages.png")
    figure_3_delta(assignments, FIGURES / "figure_3_template_effect_delta.png")
    figure_4_stacked(assignments, FIGURES / "figure_4_stacked_cluster_composition.png")
    figure_5_oligomer(assignments, FIGURES / "figure_5_oligomer_size_vs_cluster.png")
    figure_6_dendrogram(assignments, FIGURES / "figure_6_method_similarity_dendrogram.png")
    figure_7_boxplot(assignments, FIGURES / "figure_7_qtmscore_quality_boxplot.png")
    figure_8_by_mer(assignments, FIGURES / "figure_8_qtmscore_by_oligomer_size.png")
    figure_9_targets(assignments, FIGURES / "figure_9_top_pdb_representatives.png")
    figure_10_global_all_vs_all(qsym, FIGURES / "figure_10_all_vs_all_sym_qtm_heatmap.png")



# =============================================================================
# ANALYSIS DRIVER
# =============================================================================

def analyse() -> None:
    if not INVENTORY_TSV.is_file():
        raise FileNotFoundError(f"Missing inventory: {INVENTORY_TSV}; run --prepare first.")
    if not PRED_VS_EXP_TSV.is_file() or not ALL_VS_ALL_TSV.is_file():
        raise FileNotFoundError("Foldseek outputs are missing; run --run-foldseek first.")

    print("=" * 84)
    print("BUILDING MODEL-LEVEL EXPERIMENTAL ASSIGNMENTS")
    print("=" * 84)
    assignments = build_assignments()
    if len(assignments) != EXPECTED_PREDICTIONS:
        raise RuntimeError(f"Expected {EXPECTED_PREDICTIONS} assignment rows, found {len(assignments)}")
    switches = build_template_switching(assignments)
    if len(switches) != EXPECTED_SWITCHES:
        raise RuntimeError(f"Expected {EXPECTED_SWITCHES} paired template/no-template comparisons, found {len(switches)}")
    make_summary(assignments, switches)

    print("\n" + "=" * 84)
    print(f"BUILDING {EXPECTED_TOTAL} x {EXPECTED_TOTAL} ALL-vs-ALL MATRICES")
    print("=" * 84)
    qsym, _ = build_symmetric_matrices()

    print("\n" + "=" * 84)
    print("CREATING FINAL FIGURES")
    print("=" * 84)
    create_figures(assignments, qsym)

    print("\nFINAL ANALYSIS COMPLETE")
    print(f"Assignments: {ASSIGN_TSV}")
    print(f"Template switching: {SWITCH_TSV}")
    print(f"Matrices: {MATRIX_DIR}")
    print(f"Figures:  {FIGURES}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Peptide-separated Aβ representative-chain Foldseek pipeline; methodology unchanged from the combined run")
    actions = p.add_mutually_exclusive_group()
    actions.add_argument("--prepare", action="store_true", help="Extract/copy the peptide-specific representative-chain inputs.")
    actions.add_argument("--run-foldseek", action="store_true", help="Run peptide-specific predicted-vs-experimental and all-vs-all Foldseek.")
    actions.add_argument("--analyse", action="store_true", help="Build assignments, matrices, tables and figures from Foldseek outputs.")
    actions.add_argument("--all", action="store_true", help="Prepare, run Foldseek, then analyse in one invocation.")
    p.add_argument("--peptide", choices=["ab40", "ab42"], required=True, help="Run the unchanged workflow for Aβ40 or Aβ42 only.")
    p.add_argument("--force", action="store_true", help="Overwrite/rebuild derived files inside the selected peptide output folders only.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.all:
        prepare_representatives(args.force)
        run_foldseek(args.force)
        analyse()
    elif args.prepare:
        prepare_representatives(args.force)
    elif args.run_foldseek:
        run_foldseek(args.force)
    elif args.analyse:
        analyse()
    else:
        source_preflight(require_foldseek=False)
        print("READ-ONLY PREFLIGHT ONLY. No project files or derived outputs were changed.")
        print("Next: run with --prepare after reviewing the counts above.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
