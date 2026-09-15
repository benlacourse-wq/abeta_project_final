#!/usr/bin/env python3
"""
Build an ECOD/CATH-inspired hierarchical structural taxonomy for experimental Aβ40/Aβ42.

Core design
-----------
* Experimental structures define the taxonomy.
* Aβ40 (33 representatives) and Aβ42 (46 representatives) are analysed separately.
* Symmetric qTM is the PRIMARY structural similarity used for hierarchical clustering.
* Clustering uses average linkage on distance = 1 - symmetric qTM.
* Metadata annotate/interrogate the structure-defined hierarchy; metadata never drives clustering.
* Contact-Jaccard, RMSD and alnTM are companion validation matrices using the SAME ordering.
* Prediction models are projected onto the finished experimental taxonomy after family/polymorph
  cut heights have been selected from diagnostics.
* No source files are modified. Everything is written to new analysis/figure directories.

The script intentionally separates two stages:
  1) metadata repair + hierarchy diagnostics (no final cut imposed), and
  2) final taxonomy build after explicit family/polymorph cut heights are supplied.

Examples
--------
Read-only-ish preflight (creates nothing):
    python build_abeta_ecod_cath_hierarchy.py --preflight

Repair/enrich metadata and generate hierarchy diagnostics:
    python build_abeta_ecod_cath_hierarchy.py --diagnostics

After reviewing the diagnostic tables/figures, build final taxonomy (example cut values ONLY):
    python build_abeta_ecod_cath_hierarchy.py --build-taxonomy \
        --ab40-family-cut 0.45 --ab40-polymorph-cut 0.30 \
        --ab42-family-cut 0.42 --ab42-polymorph-cut 0.28

IMPORTANT: do not copy the example cut values without reviewing the diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from matplotlib import gridspec
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import chi2_contingency, kruskal


# ======================================================================================
# Paths / constants
# ======================================================================================

DEFAULT_PROJECT = Path("/home/benla/abeta_project_final")

PEPTIDE_INFO = {
    "ab40": {"meta": "Abeta40", "label": "Aβ40", "length": 40, "n_exp": 33, "prefix": "A40"},
    "ab42": {"meta": "Abeta42", "label": "Aβ42", "length": 42, "n_exp": 46, "prefix": "A42"},
}

# Explicit source signals. These are deliberately conservative and are only used to derive
# broad source classes from the supplied Amyloid Explorer metadata.
ANIMAL_TERMS = (
    "mouse", "murine", "tg-swdi", "tg swdi", "arte10", "app23", "app/ps1",
    "appnl-g-f", "app(nl-g-f)", "tg-apparcswe", "tgappswe", "tg-app",
)

HUMAN_EXPLICIT_TERMS = (
    "human brain", "patient", "down syndrome", "brain-derived",
)

METHOD_MAP = {
    "ELECTRON MICROSCOPY": "Electron microscopy",
    "SOLID-STATE NMR": "Solid-state NMR",
    "SOLUTION NMR": "Solution NMR",
}

SOURCE_PALETTE = {
    "Human-derived": "#4C78A8",
    "Animal-derived": "#F58518",
    "In vitro / synthetic / unclear": "#B8B8B8",
}
METHOD_PALETTE = {
    "Electron microscopy": "#59A14F",
    "Solid-state NMR": "#E15759",
    "Solution NMR": "#B279A2",
}
MOD_PALETTE = {
    "WT / unmodified": "#D9D9D9",
    "Modified / mutant": "#E69F00",
}
DISEASE_PALETTE = {
    "AD": "#D62728",
    "CAA": "#9467BD",
    "AD + CAA": "#8C564B",
    "Down syndrome": "#17BECF",
    "Other neurological": "#BCBD22",
    "Not reported / non-human": "#D9D9D9",
}
TISSUE_PALETTE = {
    "Cortical / parenchymal brain": "#1F77B4",
    "Meningeal / vascular": "#9467BD",
    "Animal model": "#FF7F0E",
    "Human tissue - unspecified": "#8C564B",
    "In vitro / not applicable": "#D9D9D9",
}


# ======================================================================================
# Generic helpers
# ======================================================================================

def banner(text: str) -> None:
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)


def normalise_structure_id(value: object) -> str:
    """Normalise a structure identifier while preserving chain id case where possible."""
    text = Path(str(value).strip()).name
    for ext in (".pdb", ".cif", ".mmcif"):
        if text.lower().endswith(ext):
            text = text[: -len(ext)]
            break
    m = re.match(r"^([A-Za-z0-9]{4})_chain_(.+)$", text, flags=re.I)
    if m:
        return f"{m.group(1).upper()}_chain_{m.group(2)}"
    if re.match(r"^[A-Za-z0-9]{4}$", text):
        return text.upper()
    return text


def pdb_from_structure_id(value: object) -> str:
    text = normalise_structure_id(value)
    m = re.match(r"^([A-Za-z0-9]{4})", text)
    return m.group(1).upper() if m else text[:4].upper()


def parse_chain_from_representative(value: object) -> str | None:
    text = Path(str(value)).name
    m = re.search(r"_chain_(.+?)\.(?:cif|pdb|mmcif)$", text, flags=re.I)
    return m.group(1) if m else None


def parse_residue_numbers(value: object) -> list[int]:
    if pd.isna(value):
        return []
    out = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(float(token)))
        except ValueError:
            continue
    return out


def safe_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def natural_cluster_key(value: object) -> tuple:
    text = str(value)
    nums = [int(x) for x in re.findall(r"\d+", text)]
    return (*nums, text)


def load_square_matrix(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0)
    df.index = [normalise_structure_id(x) for x in df.index]
    df.columns = [normalise_structure_id(x) for x in df.columns]
    if df.index.duplicated().any() or pd.Index(df.columns).duplicated().any():
        raise ValueError(f"Duplicate identifiers after normalisation in {path}")
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def symmetrise(df: pd.DataFrame, diagonal: float | None = None) -> pd.DataFrame:
    common = [x for x in df.index if x in df.columns]
    d = df.loc[common, common].copy()
    arr = d.to_numpy(dtype=float, copy=True)
    arr = np.nanmean(np.stack([arr, arr.T]), axis=0)
    if diagonal is not None:
        np.fill_diagonal(arr, diagonal)
    return pd.DataFrame(arr, index=common, columns=common)


def silhouette_from_distance(dist: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score for a precomputed distance matrix; singleton silhouettes are 0."""
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    n = len(labels)
    if len(uniq) < 2 or len(uniq) >= n:
        return float("nan")
    values = []
    for i in range(n):
        own = labels[i]
        own_idx = np.where(labels == own)[0]
        own_idx = own_idx[own_idx != i]
        if len(own_idx) == 0:
            values.append(0.0)
            continue
        a = float(np.mean(dist[i, own_idx]))
        b = float("inf")
        for other in uniq:
            if other == own:
                continue
            idx = np.where(labels == other)[0]
            if len(idx):
                b = min(b, float(np.mean(dist[i, idx])))
        denom = max(a, b)
        values.append(0.0 if denom <= 0 or not np.isfinite(denom) else (b - a) / denom)
    return float(np.mean(values))


def stable_relabel(labels: np.ndarray, leaf_order: np.ndarray, prefix: str) -> np.ndarray:
    """Relabel clusters by first appearance in dendrogram order for stable readable IDs."""
    first_pos = {}
    for pos, idx in enumerate(leaf_order):
        lab = int(labels[idx])
        first_pos.setdefault(lab, pos)
    ordered = sorted(first_pos, key=first_pos.get)
    mapping = {old: f"{prefix}{i:02d}" for i, old in enumerate(ordered, start=1)}
    return np.array([mapping[int(x)] for x in labels], dtype=object)


# ======================================================================================
# Metadata repair / enrichment
# ======================================================================================

def classify_source(row: pd.Series) -> tuple[str, str, str]:
    desc = safe_str(row.get("Description"))
    tissue = safe_str(row.get("Tissue"))
    text = f"{desc} {tissue}".lower()
    raw_tick = bool(safe_str(row.get("From Patient")))

    animal_hits = [term for term in ANIMAL_TERMS if term in text]
    if animal_hits:
        return (
            "Animal-derived",
            "Animal-model brain/tissue",
            "animal signal: " + ", ".join(animal_hits[:3]),
        )

    explicit_human = [term for term in HUMAN_EXPLICIT_TERMS if term in text]
    if raw_tick or explicit_human:
        seeded = "seed" in text
        subtype = "Human-derived seed / seeded preparation" if seeded else "Human brain/tissue"
        basis = "Amyloid Explorer From Patient flag" if raw_tick else "explicit description: " + ", ".join(explicit_human)
        return "Human-derived", subtype, basis

    return "In vitro / synthetic / unclear", "In vitro / synthetic / not source-annotated", "no human/animal source signal"


def normalise_disease(row: pd.Series) -> str:
    if row["source_class"] != "Human-derived":
        return "Not reported / non-human"
    raw = safe_str(row.get("Disease"))
    desc = safe_str(row.get("Description"))
    text = f"{raw} {desc}".lower()
    if "down syndrome" in text:
        return "Down syndrome"
    has_caa = "caa" in text or "cerebral amyloid angiopathy" in text or "vascular amyloidosis" in text
    has_ad = bool(re.search(r"\balzheimer|\bad\d*\b|familial ad|sporadic ad|dominantly inherited ad", text))
    if has_caa and has_ad:
        return "AD + CAA"
    if has_caa:
        return "CAA"
    if has_ad:
        return "AD"
    if "als/pdc" in text or "guam" in text:
        return "Other neurological"
    return "Not reported / non-human"


def normalise_tissue(row: pd.Series) -> str:
    if row["source_class"] == "Animal-derived":
        return "Animal model"
    if row["source_class"] != "Human-derived":
        return "In vitro / not applicable"
    text = f"{safe_str(row.get('Tissue'))} {safe_str(row.get('Description'))}".lower()
    if any(x in text for x in ("leptomening", "meningeal", "vessel", "vascular")):
        return "Meningeal / vascular"
    if any(x in text for x in ("cortex", "cortical", "brain lobe", "frontal", "parietal", "occipital", "human brain", "brain-derived")):
        return "Cortical / parenchymal brain"
    return "Human tissue - unspecified"


def normalise_modification(value: object) -> tuple[str, str]:
    raw = safe_str(value)
    if not raw:
        return "WT / unmodified", "WT / unmodified"
    u = raw.upper().replace("Β", "BETA")
    if "UPPSALA" in u:
        exact = "Uppsala mutation"
    elif "GALNAC" in u or "GLYCOSYL" in u:
        exact = "Y10 glycosylation"
    elif "D-ASPARTATE 7" in u:
        exact = "D-Asp7 / D-Asp23"
    elif "D-ASPARTATE 23" in u:
        exact = "D-Asp23"
    elif raw.upper() == "OSAKA":
        exact = "Osaka mutation"
    else:
        exact = raw
    return "Modified / mutant", exact


def build_enriched_metadata(project: Path, write_outputs: bool = True) -> tuple[pd.DataFrame, str]:
    meta_dir = project / "experimental" / "metadata"
    clean_path = meta_dir / "abeta_metadata_clean.tsv"
    raw40_path = meta_dir / "AB40_Amyloid_Explorer.csv"
    raw42_path = meta_dir / "ABETA_42_Amyloid_Explorer.csv"
    chain_path = meta_dir / "chain_summary.tsv"
    decisions_path = meta_dir / "representative_chain_decisions.tsv"

    for p in (clean_path, raw40_path, raw42_path, chain_path, decisions_path):
        if not p.is_file():
            raise FileNotFoundError(f"Required metadata source not found: {p}")

    clean = pd.read_csv(clean_path, sep="\t")
    raw40 = pd.read_csv(raw40_path)
    raw42 = pd.read_csv(raw42_path)
    raw = pd.concat([raw40, raw42], ignore_index=True)
    chain = pd.read_csv(chain_path, sep="\t")
    decisions = pd.read_csv(decisions_path, sep="\t")

    clean = clean.copy()
    clean["pdb_id"] = clean["pdb_id"].astype(str).str.upper()
    clean["structure"] = clean["structure"].map(normalise_structure_id)
    raw = raw.copy()
    raw["pdb_id"] = raw["PDB ID"].astype(str).str.upper()
    decisions = decisions.copy()
    decisions["pdb_id"] = decisions["pdb_id"].astype(str).str.upper()
    chain = chain.copy()
    chain["pdb_id"] = chain["pdb_id"].astype(str).str.upper()
    chain["chain_id"] = chain["chain_id"].astype(str)

    # Strict final dataset checks.
    if len(clean) != 79 or clean["pdb_id"].nunique() != 79:
        raise ValueError(f"Expected 79 unique final experimental representatives; found rows={len(clean)}, unique PDBs={clean['pdb_id'].nunique()}")
    counts = clean["peptide"].value_counts().to_dict()
    if counts.get("Abeta40", 0) != 33 or counts.get("Abeta42", 0) != 46:
        raise ValueError(f"Final peptide counts are not 33 Aβ40 / 46 Aβ42: {counts}")

    raw_final = raw[raw["pdb_id"].isin(clean["pdb_id"])].copy()
    if raw_final["pdb_id"].nunique() != 79:
        missing = sorted(set(clean["pdb_id"]) - set(raw_final["pdb_id"]))
        raise ValueError("Raw Amyloid Explorer metadata missing final PDB IDs: " + ", ".join(missing))
    excluded_raw = sorted(set(raw["pdb_id"]) - set(clean["pdb_id"]))

    # Merge raw fields with explicit prefixes; preserve original clean fields too.
    raw_keep = [
        "pdb_id", "Updated", "Modified", "Description", "Residues (Length)", "Dep. Date",
        "Method", "Resolution", "Authors", "From Patient", "Disease", "Tissue",
        "Role (P/F)", "BMRB", "EMDB", "PubMed", "UniProt",
    ]
    raw_named = raw_final[raw_keep].copy().rename(columns={
        "Updated": "raw_updated",
        "Modified": "raw_modified",
        "Description": "raw_description",
        "Residues (Length)": "raw_residues_length",
        "Dep. Date": "raw_deposition_date",
        "Method": "raw_method",
        "Resolution": "raw_resolution",
        "Authors": "raw_authors",
        "From Patient": "raw_from_patient",
        "Disease": "raw_disease",
        "Tissue": "raw_tissue",
        "Role (P/F)": "raw_role",
        "BMRB": "raw_bmrb",
        "EMDB": "raw_emdb",
        "PubMed": "raw_pubmed",
        "UniProt": "raw_uniprot",
    })
    enriched = clean.merge(raw_named, on="pdb_id", how="left", validate="one_to_one")

    dec = decisions[["pdb_id", "chosen_representative", "n_chains_available", "reason"]].copy()
    enriched = enriched.merge(dec, on="pdb_id", how="left", validate="one_to_one")
    if enriched["chosen_representative"].isna().any():
        raise ValueError("Some final representatives have no representative-chain decision.")
    enriched["chosen_chain_id"] = enriched["chosen_representative"].map(parse_chain_from_representative)

    # Reconstruct actual coordinate residue coverage from chain_summary.
    coord_rows = []
    for _, r in enriched.iterrows():
        pdb = r["pdb_id"]
        cid = str(r["chosen_chain_id"])
        sub = chain[(chain["pdb_id"] == pdb) & (chain["chain_id"] == cid) & (chain["error"].isna())].copy()
        if sub.empty:
            raise ValueError(f"No valid chain_summary row for final representative {pdb} chain {cid}")
        if "model_id" in sub.columns:
            sub = sub.sort_values("model_id", na_position="last")
        rr = sub.iloc[0]
        nums = parse_residue_numbers(rr["residue_numbers"])
        if not nums:
            raise ValueError(f"No residue numbers for {pdb} chain {cid}")
        nums = sorted(dict.fromkeys(nums))
        peptide_length = 40 if r["peptide"] == "Abeta40" else 42
        internal_missing = [x for x in range(min(nums), max(nums) + 1) if x not in nums]
        absent_vs_reference = [x for x in range(1, peptide_length + 1) if x not in nums]
        coord_rows.append({
            "pdb_id": pdb,
            "modelled_residue_start": min(nums),
            "modelled_residue_end": max(nums),
            "n_modelled_residues": len(nums),
            "modelled_residue_numbers": ",".join(map(str, nums)),
            "internal_absent_positions": ",".join(map(str, internal_missing)) if internal_missing else "",
            "n_internal_absent_positions": len(internal_missing),
            "absent_positions_vs_reference": ",".join(map(str, absent_vs_reference)) if absent_vs_reference else "",
            "n_absent_positions_vs_reference": len(absent_vs_reference),
            "reference_peptide_length": peptide_length,
            "coordinate_fraction_of_reference": len(nums) / peptide_length,
        })
    coords = pd.DataFrame(coord_rows)
    enriched = enriched.merge(coords, on="pdb_id", how="left", validate="one_to_one")

    # Derived cleaned categorical metadata.
    def source_apply(r: pd.Series) -> pd.Series:
        tmp = pd.Series({
            "Description": r.get("raw_description"),
            "Tissue": r.get("raw_tissue"),
            "From Patient": r.get("raw_from_patient"),
        })
        a, b, c = classify_source(tmp)
        return pd.Series([a, b, c])

    enriched[["source_class", "source_subtype", "source_class_basis"]] = enriched.apply(source_apply, axis=1)
    enriched["method_group"] = enriched["raw_method"].map(METHOD_MAP).fillna(enriched["method"])
    enriched["modification_status"] = enriched["raw_modified"].map(lambda x: normalise_modification(x)[0])
    enriched["modification_detail"] = enriched["raw_modified"].map(lambda x: normalise_modification(x)[1])
    enriched["disease_group"] = enriched.apply(lambda r: normalise_disease(pd.Series({
        "source_class": r["source_class"], "Disease": r["raw_disease"], "Description": r["raw_description"]
    })), axis=1)
    enriched["tissue_group"] = enriched.apply(lambda r: normalise_tissue(pd.Series({
        "source_class": r["source_class"], "Tissue": r["raw_tissue"], "Description": r["raw_description"]
    })), axis=1)
    enriched["em_resolution_A"] = np.where(
        enriched["method_group"].eq("Electron microscopy"),
        pd.to_numeric(enriched["raw_resolution"], errors="coerce"),
        np.nan,
    )
    enriched["legacy_qtm_cluster"] = enriched["cluster"].astype(str)

    # A concise corrected coverage label.
    def coverage_label(r: pd.Series) -> str:
        core = f"{int(r.modelled_residue_start)}-{int(r.modelled_residue_end)}"
        if int(r.n_internal_absent_positions) > 0:
            core += f"; internal absent: {r.internal_absent_positions}"
        return f"{core}; {int(r.n_modelled_residues)}/{int(r.reference_peptide_length)} ({100*r.coordinate_fraction_of_reference:.1f}%)"
    enriched["coordinate_coverage_label"] = enriched.apply(coverage_label, axis=1)

    # Compare legacy residue string with actual coordinate summary, purely as QC.
    def parse_legacy_count(x: object) -> float:
        m = re.search(r"\(([-\d]+)\)", safe_str(x))
        return float(m.group(1)) if m else np.nan
    enriched["legacy_reported_count"] = enriched["residues_present"].map(parse_legacy_count)
    enriched["legacy_count_matches_actual"] = enriched["legacy_reported_count"].eq(enriched["n_modelled_residues"])

    # Order columns: cleaned analysis fields first, raw/provenance later.
    first_cols = [
        "structure", "pdb_id", "peptide", "legacy_qtm_cluster",
        "chosen_representative", "chosen_chain_id", "n_chains_available",
        "modelled_residue_start", "modelled_residue_end", "n_modelled_residues",
        "reference_peptide_length", "coordinate_fraction_of_reference",
        "n_internal_absent_positions", "internal_absent_positions",
        "n_absent_positions_vs_reference", "absent_positions_vs_reference",
        "coordinate_coverage_label",
        "source_class", "source_subtype", "source_class_basis",
        "method_group", "em_resolution_A",
        "modification_status", "modification_detail",
        "disease_group", "tissue_group",
    ]
    remaining = [c for c in enriched.columns if c not in first_cols]
    enriched = enriched[first_cols + remaining]

    # Report.
    source_counts = enriched["source_class"].value_counts().to_dict()
    mod_counts = enriched["modification_status"].value_counts().to_dict()
    incorrect_legacy = int((~enriched["legacy_count_matches_actual"]).sum())
    report_lines = [
        "Aβ EXPERIMENTAL METADATA REPAIR REPORT",
        "=" * 72,
        f"Final representatives: {len(enriched)} (Aβ40={(enriched.peptide=='Abeta40').sum()}, Aβ42={(enriched.peptide=='Abeta42').sum()})",
        f"Raw Amyloid Explorer rows: {len(raw)}",
        f"Raw entries excluded because they are not in the final 79: {', '.join(excluded_raw) if excluded_raw else 'none'}",
        "",
        "SOURCE CLASS (derived conservatively from supplied metadata)",
        *[f"  {k}: {v}" for k, v in source_counts.items()],
        "",
        "MODIFICATION STATUS",
        *[f"  {k}: {v}" for k, v in mod_counts.items()],
        "",
        f"Legacy residues_present count disagrees with chain_summary actual coordinate count: {incorrect_legacy}/{len(enriched)}",
        "The final taxonomy uses chain_summary-derived coordinate residue coverage, not the legacy residues_present count.",
        "",
        "IMPORTANT SOURCE NOTE",
        "The Amyloid Explorer 'From Patient' tick is retained as raw provenance but is NOT used directly as a binary patient-derived variable.",
        "Animal-model entries with that tick are classified as Animal-derived. Explicit patient descriptions can recover human-derived entries even if the tick is blank.",
        "",
        "IMPORTANT RESOLUTION NOTE",
        "em_resolution_A is populated only for electron-microscopy structures; NMR numeric values are retained only in raw_resolution and are not treated as comparable EM resolution.",
    ]
    report = "\n".join(report_lines) + "\n"

    if write_outputs:
        out = project / "analysis" / "ecod_cath_hierarchy" / "metadata"
        out.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(out / "abeta_experimental_metadata_enriched_79.tsv", sep="\t", index=False)
        enriched[enriched["peptide"] == "Abeta40"].to_csv(out / "ab40_experimental_metadata_enriched_33.tsv", sep="\t", index=False)
        enriched[enriched["peptide"] == "Abeta42"].to_csv(out / "ab42_experimental_metadata_enriched_46.tsv", sep="\t", index=False)
        (out / "metadata_repair_report.txt").write_text(report, encoding="utf-8")
        # Explicit legacy discrepancy audit.
        enriched[[
            "structure", "pdb_id", "peptide", "residues_present", "legacy_reported_count",
            "modelled_residue_start", "modelled_residue_end", "n_modelled_residues",
            "coordinate_coverage_label", "legacy_count_matches_actual",
        ]].to_csv(out / "legacy_residue_metadata_vs_chain_summary.tsv", sep="\t", index=False)

    return enriched, report


# ======================================================================================
# Structural matrices / hierarchy diagnostics
# ======================================================================================

def get_experimental_matrices(project: Path, metadata: pd.DataFrame, peptide_key: str) -> dict[str, pd.DataFrame]:
    info = PEPTIDE_INFO[peptide_key]
    fold = project / "experimental" / "foldseek" / "matrices"
    qtm = load_square_matrix(fold / "qtmscore_matrix.csv")
    alntm = load_square_matrix(fold / "alntmscore_matrix.csv") if (fold / "alntmscore_matrix.csv").is_file() else None
    rmsd = load_square_matrix(fold / "rmsd_matrix.csv") if (fold / "rmsd_matrix.csv").is_file() else None

    submeta = metadata[metadata["peptide"] == info["meta"]].copy()
    ids = submeta["structure"].map(normalise_structure_id).tolist()
    if len(ids) != info["n_exp"]:
        raise ValueError(f"Expected {info['n_exp']} {info['label']} metadata structures; found {len(ids)}")

    missing_q = [x for x in ids if x not in qtm.index or x not in qtm.columns]
    if missing_q:
        raise ValueError(f"qTM matrix missing {len(missing_q)} {info['label']} final structures: {missing_q}")

    q = symmetrise(qtm.loc[ids, ids], diagonal=1.0)
    out = {"qtm": q}
    if alntm is not None:
        missing = [x for x in ids if x not in alntm.index or x not in alntm.columns]
        if not missing:
            out["alntm"] = symmetrise(alntm.loc[ids, ids], diagonal=1.0)
    if rmsd is not None:
        missing = [x for x in ids if x not in rmsd.index or x not in rmsd.columns]
        if not missing:
            r = symmetrise(rmsd.loc[ids, ids], diagonal=0.0)
            out["rmsd"] = r
    return out


def hierarchy_from_qtm(qtm: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sim = np.clip(qtm.to_numpy(dtype=float, copy=True), 0.0, 1.0)
    dist = 1.0 - sim
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average", optimal_ordering=True)
    order = leaves_list(Z)
    return dist, Z, order


def scan_cut_heights(qtm: pd.DataFrame, dist: np.ndarray, Z: np.ndarray) -> pd.DataFrame:
    off = dist[~np.eye(dist.shape[0], dtype=bool)]
    low = max(0.02, float(np.nanpercentile(off, 2)))
    high = min(0.98, float(np.nanpercentile(off, 98)))
    cuts = np.unique(np.round(np.linspace(low, high, 151), 4))
    sim = qtm.to_numpy(dtype=float)
    rows = []
    for cut in cuts:
        labels = fcluster(Z, t=float(cut), criterion="distance")
        counts = Counter(labels)
        same = labels[:, None] == labels[None, :]
        offdiag = ~np.eye(len(labels), dtype=bool)
        within_mask = same & offdiag
        between_mask = (~same) & offdiag
        within = float(np.nanmean(sim[within_mask])) if np.any(within_mask) else np.nan
        between = float(np.nanmean(sim[between_mask])) if np.any(between_mask) else np.nan
        rows.append({
            "cut_distance": float(cut),
            "equivalent_qtm_similarity": 1.0 - float(cut),
            "n_clusters": len(counts),
            "n_singletons": sum(v == 1 for v in counts.values()),
            "singleton_fraction": sum(v == 1 for v in counts.values()) / len(counts),
            "largest_cluster_n": max(counts.values()),
            "mean_within_qtm": within,
            "mean_between_qtm": between,
            "within_minus_between_qtm": within - between if np.isfinite(within) and np.isfinite(between) else np.nan,
            "silhouette": silhouette_from_distance(dist, labels),
        })
    return pd.DataFrame(rows)


def candidate_cut_table(scan: pd.DataFrame) -> pd.DataFrame:
    candidates = []
    # Broad family candidates: deliberately broad structural groupings.
    fam = scan[(scan.n_clusters >= 3) & (scan.n_clusters <= 12) & (scan.singleton_fraction <= 0.50)].copy()
    if not fam.empty:
        fam = fam.sort_values(["silhouette", "within_minus_between_qtm"], ascending=False).head(12)
        fam.insert(0, "candidate_level", "family")
        candidates.append(fam)
    # Finer polymorph candidates. Still diagnostic only, not an automatic decision.
    pol = scan[(scan.n_clusters >= 5) & (scan.n_clusters <= 30) & (scan.singleton_fraction <= 0.75)].copy()
    if not pol.empty:
        pol = pol.sort_values(["silhouette", "within_minus_between_qtm"], ascending=False).head(15)
        pol.insert(0, "candidate_level", "polymorph")
        candidates.append(pol)
    return pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()


def plot_dendrogram(Z: np.ndarray, ids: list[str], title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    dendrogram(Z, labels=[pdb_from_structure_id(x) for x in ids], leaf_rotation=90, leaf_font_size=8, ax=ax)
    ax.set_ylabel("Structural distance (1 − symmetric qTM)")
    ax.set_title(title, fontsize=15, weight="bold")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cut_scan(scan: pd.DataFrame, title: str, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    x = scan["cut_distance"]
    axes[0, 0].plot(x, scan["n_clusters"])
    axes[0, 0].set_ylabel("Number of clusters")
    axes[0, 0].set_title("Cluster number")
    axes[0, 1].plot(x, scan["singleton_fraction"])
    axes[0, 1].set_ylabel("Singleton fraction")
    axes[0, 1].set_title("Singleton burden")
    axes[1, 0].plot(x, scan["silhouette"])
    axes[1, 0].set_ylabel("Silhouette score")
    axes[1, 0].set_title("Partition separation")
    axes[1, 1].plot(x, scan["within_minus_between_qtm"])
    axes[1, 1].set_ylabel("Mean within − between qTM")
    axes[1, 1].set_title("Similarity separation")
    for ax in axes.flat:
        ax.set_xlabel("Cut distance (1 − symmetric qTM)")
        ax.grid(alpha=0.2)
    fig.suptitle(title, fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_diagnostics(project: Path, metadata: pd.DataFrame) -> dict[str, dict]:
    out = project / "analysis" / "ecod_cath_hierarchy" / "diagnostics"
    figout = project / "figures" / "ecod_cath_hierarchy" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    figout.mkdir(parents=True, exist_ok=True)
    results = {}

    for key, info in PEPTIDE_INFO.items():
        banner(f"{info['label']} HIERARCHY DIAGNOSTICS")
        mats = get_experimental_matrices(project, metadata, key)
        qtm = mats["qtm"]
        dist, Z, order = hierarchy_from_qtm(qtm)
        scan = scan_cut_heights(qtm, dist, Z)
        candidates = candidate_cut_table(scan)

        qtm.to_csv(out / f"{key}_symmetric_qtm_{len(qtm)}x{len(qtm)}.csv", float_format="%.6f")
        pd.DataFrame(dist, index=qtm.index, columns=qtm.columns).to_csv(
            out / f"{key}_qtm_distance_{len(qtm)}x{len(qtm)}.csv", float_format="%.6f"
        )
        pd.DataFrame(Z, columns=["left", "right", "distance", "n_members"]).to_csv(
            out / f"{key}_average_linkage.tsv", sep="\t", index=False
        )
        scan.to_csv(out / f"{key}_cut_height_scan.tsv", sep="\t", index=False)
        candidates.to_csv(out / f"{key}_candidate_cuts.tsv", sep="\t", index=False)
        pd.DataFrame({
            "dendrogram_position": np.arange(len(order)),
            "structure": qtm.index[order],
            "pdb_id": [pdb_from_structure_id(x) for x in qtm.index[order]],
        }).to_csv(out / f"{key}_dendrogram_order.tsv", sep="\t", index=False)

        plot_dendrogram(
            Z, qtm.index.tolist(),
            f"{info['label']} experimental structural hierarchy — average linkage on symmetric qTM",
            figout / f"{key}_qtm_dendrogram.png",
        )
        plot_cut_scan(
            scan,
            f"{info['label']} hierarchy cut diagnostics",
            figout / f"{key}_cut_diagnostics.png",
        )

        print(f"Structures: {len(qtm)}")
        print(f"qTM matrix: {qtm.shape[0]} × {qtm.shape[1]}")
        print(f"Average-linkage terminal merge distance: {Z[-1, 2]:.4f}")
        if not candidates.empty:
            show = candidates[[
                "candidate_level", "cut_distance", "equivalent_qtm_similarity", "n_clusters",
                "n_singletons", "singleton_fraction", "silhouette", "within_minus_between_qtm",
            ]].head(12)
            print("\nTop diagnostic candidate cuts (NOT final assignments):")
            print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        results[key] = {"mats": mats, "dist": dist, "Z": Z, "order": order, "scan": scan, "candidates": candidates}

    return results


# ======================================================================================
# Companion contact matrix from actual representative coordinates
# ======================================================================================

def load_ca_contact_set(pdb_path: Path, cutoff: float = 8.0, min_seq_sep: int = 3) -> tuple[set[int], set[tuple[int, int]]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = next(structure.get_models())
    chains = list(model.get_chains())
    if not chains:
        raise ValueError(f"No chains in {pdb_path}")
    chain = chains[0]
    coords: dict[int, np.ndarray] = {}
    for res in chain.get_residues():
        het, resseq, icode = res.id
        if het.strip():
            continue
        if "CA" not in res:
            continue
        coords[int(resseq)] = np.asarray(res["CA"].coord, dtype=float)
    residues = set(coords)
    contacts: set[tuple[int, int]] = set()
    nums = sorted(coords)
    for a_i, i in enumerate(nums):
        for j in nums[a_i + 1:]:
            if abs(j - i) < min_seq_sep:
                continue
            if float(np.linalg.norm(coords[i] - coords[j])) <= cutoff:
                contacts.add((i, j))
    return residues, contacts


def compute_contact_jaccard_matrix(project: Path, metadata: pd.DataFrame, peptide_key: str, ordered_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    reps = project / "experimental" / "structures" / "representatives_pdb_79"
    sub = metadata.set_index("structure")
    cache = {}
    for sid in ordered_ids:
        if sid not in sub.index:
            raise KeyError(f"Metadata missing structure {sid}")
        p = reps / f"{sid}.pdb"
        if not p.is_file():
            raise FileNotFoundError(p)
        cache[sid] = load_ca_contact_set(p)

    n = len(ordered_ids)
    jac = np.full((n, n), np.nan, dtype=float)
    common_frac = np.full((n, n), np.nan, dtype=float)
    pep_len = PEPTIDE_INFO[peptide_key]["length"]
    for i, a in enumerate(ordered_ids):
        ra, ca = cache[a]
        jac[i, i] = 1.0
        common_frac[i, i] = len(ra) / pep_len
        for j in range(i + 1, n):
            b = ordered_ids[j]
            rb, cb = cache[b]
            common = ra & rb
            common_frac[i, j] = common_frac[j, i] = len(common) / pep_len
            if len(common) < 2:
                continue
            ca_r = {p for p in ca if p[0] in common and p[1] in common}
            cb_r = {p for p in cb if p[0] in common and p[1] in common}
            union = ca_r | cb_r
            if not union:
                value = np.nan
            else:
                value = len(ca_r & cb_r) / len(union)
            jac[i, j] = jac[j, i] = value
    return (
        pd.DataFrame(jac, index=ordered_ids, columns=ordered_ids),
        pd.DataFrame(common_frac, index=ordered_ids, columns=ordered_ids),
    )


# ======================================================================================
# Final taxonomy build
# ======================================================================================

def build_taxonomy_assignments(
    metadata: pd.DataFrame,
    peptide_key: str,
    qtm: pd.DataFrame,
    Z: np.ndarray,
    order: np.ndarray,
    family_cut: float,
    polymorph_cut: float,
) -> pd.DataFrame:
    if not (0 < polymorph_cut < family_cut < 1.0):
        raise ValueError(
            f"For {peptide_key}, require 0 < polymorph_cut < family_cut < 1 because polymorph is the finer level. "
            f"Got family={family_cut}, polymorph={polymorph_cut}."
        )
    info = PEPTIDE_INFO[peptide_key]
    fam_raw = fcluster(Z, t=family_cut, criterion="distance")
    pol_raw = fcluster(Z, t=polymorph_cut, criterion="distance")
    fam_ids = stable_relabel(fam_raw, order, f"{info['prefix']}-F")

    # Polymorphs are labelled within each family and ordered by dendrogram appearance.
    leaf_pos = {int(idx): pos for pos, idx in enumerate(order)}
    pol_ids = np.empty(len(pol_raw), dtype=object)
    for fam in sorted(set(fam_ids), key=natural_cluster_key):
        idxs = np.where(fam_ids == fam)[0]
        raw_values = sorted(set(pol_raw[idxs]), key=lambda lab: min(leaf_pos[int(i)] for i in idxs if pol_raw[i] == lab))
        pmap = {lab: f"{fam}-P{k:02d}" for k, lab in enumerate(raw_values, start=1)}
        for i in idxs:
            pol_ids[i] = pmap[pol_raw[i]]

    base = pd.DataFrame({
        "structure": qtm.index,
        "family": fam_ids,
        "polymorph": pol_ids,
    })
    base["dendrogram_position"] = base["structure"].map({qtm.index[idx]: pos for pos, idx in enumerate(order)})
    submeta = metadata[metadata["peptide"] == info["meta"]].copy()
    tax = base.merge(submeta, on="structure", how="left", validate="one_to_one")
    return tax.sort_values("dendrogram_position").reset_index(drop=True)


def categorical_palette(values: Iterable[str], cmap_name: str = "tab20") -> dict[str, tuple]:
    values = list(dict.fromkeys(map(str, values)))
    cmap = plt.get_cmap(cmap_name, max(len(values), 1))
    return {v: cmap(i) for i, v in enumerate(values)}


def draw_annotation_strip(ax, values: list, palette: dict, label: str) -> None:
    colors = [palette.get(str(v), "#FFFFFF") for v in values]
    arr = np.arange(len(values))[None, :]
    # Each point gets its explicit RGBA via a ListedColormap.
    cmap = ListedColormap(colors)
    ax.imshow(arr, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=max(len(values)-1, 1))
    ax.set_yticks([0])
    ax.set_yticklabels([label], fontsize=8)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def cluster_boundaries(labels: list[str]) -> list[int]:
    return [i for i in range(1, len(labels)) if labels[i] != labels[i-1]]


def plot_integrated_taxonomy(
    tax: pd.DataFrame,
    qtm: pd.DataFrame,
    Z: np.ndarray,
    order: np.ndarray,
    peptide_key: str,
    family_cut: float,
    polymorph_cut: float,
    out_png: Path,
    out_pdf: Path,
) -> None:
    info = PEPTIDE_INFO[peptide_key]
    ordered = tax["structure"].tolist()
    mat = qtm.loc[ordered, ordered].to_numpy(dtype=float)
    n = len(ordered)

    # Metadata palettes.
    fam_pal = categorical_palette(tax["family"].unique(), "tab20")
    pol_pal = categorical_palette(tax["polymorph"].unique(), "gist_ncar")

    fig = plt.figure(figsize=(18, 18 if n <= 35 else 20))
    gs = gridspec.GridSpec(
        10, 3,
        height_ratios=[3.0, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 10.0],
        width_ratios=[2.8, 10.0, 1.0],
        hspace=0.05, wspace=0.05,
    )

    ax_den = fig.add_subplot(gs[0, 1])
    dendrogram(
        Z,
        no_labels=True,
        color_threshold=0,
        above_threshold_color="black",
        ax=ax_den,
    )
    # scipy dendrogram may choose the same leaf ordering but orientation x-space differs only by scaling.
    ax_den.axhline(family_cut, ls="--", lw=1.2, label=f"Family cut {family_cut:.3f}")
    ax_den.axhline(polymorph_cut, ls=":", lw=1.2, label=f"Polymorph cut {polymorph_cut:.3f}")
    ax_den.set_ylabel("1 − symmetric qTM")
    ax_den.set_xticks([])
    ax_den.legend(frameon=False, fontsize=8, loc="upper right")

    tracks = [
        ("family", fam_pal, "Family"),
        ("polymorph", pol_pal, "Polymorph"),
        ("source_class", SOURCE_PALETTE, "Source"),
        ("method_group", METHOD_PALETTE, "Method"),
        ("modification_status", MOD_PALETTE, "Modified"),
        ("disease_group", DISEASE_PALETTE, "Disease"),
        ("tissue_group", TISSUE_PALETTE, "Tissue"),
    ]
    for row_i, (col, pal, label) in enumerate(tracks, start=1):
        ax = fig.add_subplot(gs[row_i, 1])
        draw_annotation_strip(ax, tax[col].astype(str).tolist(), pal, label)

    # Continuous coordinate coverage strip.
    ax_cov = fig.add_subplot(gs[8, 1])
    cov = tax["coordinate_fraction_of_reference"].to_numpy(float)[None, :]
    ax_cov.imshow(cov, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
    ax_cov.set_yticks([0]); ax_cov.set_yticklabels(["Coord. fraction"], fontsize=8); ax_cov.set_xticks([])
    for spine in ax_cov.spines.values(): spine.set_visible(False)

    ax = fig.add_subplot(gs[9, 1])
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    labels = tax["pdb_id"].astype(str).tolist()
    ax.set_xticks(np.arange(n)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(n)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Experimental representative")
    ax.set_ylabel("Experimental representative")

    fam_bounds = cluster_boundaries(tax["family"].tolist())
    pol_bounds = cluster_boundaries(tax["polymorph"].tolist())
    for b in pol_bounds:
        ax.axhline(b - 0.5, color="white", lw=0.45, alpha=0.8)
        ax.axvline(b - 0.5, color="white", lw=0.45, alpha=0.8)
    for b in fam_bounds:
        ax.axhline(b - 0.5, color="white", lw=1.6)
        ax.axvline(b - 0.5, color="white", lw=1.6)

    cax = fig.add_subplot(gs[9, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Symmetric qTM")

    fig.suptitle(
        f"{info['label']} ECOD/CATH-inspired experimental structural taxonomy\n"
        "Hierarchy defined by average-linkage clustering of symmetric qTM; metadata are annotations only",
        fontsize=17, weight="bold", y=0.995,
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_companion_matrix(
    matrix: pd.DataFrame,
    tax: pd.DataFrame,
    title: str,
    cbar_label: str,
    out_png: Path,
    out_pdf: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> None:
    ordered = tax["structure"].tolist()
    m = matrix.loc[ordered, ordered].to_numpy(dtype=float)
    n = len(ordered)
    fig, ax = plt.subplots(figsize=(14, 12))
    masked = np.ma.masked_invalid(m)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    labels = tax["pdb_id"].astype(str).tolist()
    ax.set_xticks(np.arange(n)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(n)); ax.set_yticklabels(labels, fontsize=7)
    for b in cluster_boundaries(tax["polymorph"].tolist()):
        ax.axhline(b - 0.5, color="white", lw=0.4, alpha=0.8); ax.axvline(b - 0.5, color="white", lw=0.4, alpha=0.8)
    for b in cluster_boundaries(tax["family"].tolist()):
        ax.axhline(b - 0.5, color="white", lw=1.5); ax.axvline(b - 0.5, color="white", lw=1.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.set_label(cbar_label)
    ax.set_title(title, fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def family_polymorph_summaries(tax: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def summarise(group_col: str) -> pd.DataFrame:
        rows = []
        for name, sub in tax.groupby(group_col, sort=False):
            rows.append({
                group_col: name,
                "n_structures": len(sub),
                "pdb_ids": "; ".join(sub["pdb_id"].astype(str)),
                "source_classes": "; ".join(f"{k}:{v}" for k, v in sub["source_class"].value_counts().items()),
                "methods": "; ".join(f"{k}:{v}" for k, v in sub["method_group"].value_counts().items()),
                "modification_status": "; ".join(f"{k}:{v}" for k, v in sub["modification_status"].value_counts().items()),
                "disease_groups": "; ".join(f"{k}:{v}" for k, v in sub["disease_group"].value_counts().items()),
                "mean_coordinate_fraction": sub["coordinate_fraction_of_reference"].mean(),
            })
        return pd.DataFrame(rows)
    return summarise("family"), summarise("polymorph")


# ======================================================================================
# Prediction projection
# ======================================================================================

def project_predictions(project: Path, tax: pd.DataFrame, peptide_key: str) -> dict[str, pd.DataFrame]:
    info = PEPTIDE_INFO[peptide_key]
    assign_path = project / "analysis" / "foldseek" / f"{peptide_key}_separate" / "assignments" / "model_level_assignments_720.tsv"
    if not assign_path.is_file():
        raise FileNotFoundError(assign_path)
    pred = pd.read_csv(assign_path, sep="\t")
    if len(pred) != 720:
        raise ValueError(f"Expected 720 {info['label']} prediction assignments; found {len(pred)}")
    taxmap = tax[["pdb_id", "family", "polymorph"]].drop_duplicates("pdb_id")
    pred["best_pdb_id"] = pred["best_pdb_id"].astype(str).str.upper()
    pred = pred.merge(
        taxmap, left_on="best_pdb_id", right_on="pdb_id", how="left", validate="many_to_one"
    )
    if pred["family"].isna().any():
        missing = sorted(pred.loc[pred["family"].isna(), "best_pdb_id"].unique())
        raise ValueError(f"Prediction best-hit PDBs missing from taxonomy: {missing}")

    pred["display_method"] = pred["method"].replace({"AlphaFold": "AlphaFold 3", "Boltz": "Boltz-2", "Protenix": "Protenix-v2", "OpenFold": "OpenFold3"})
    pred["condition"] = pred["display_method"].astype(str) + " | " + pred["template_status"].replace({
        "no_templates": "no templates", "with_templates": "templates"
    }).astype(str)

    def occupancy(level: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        c = pred.groupby(["condition", level]).size().unstack(fill_value=0)
        pct = c.div(c.sum(axis=1), axis=0) * 100.0
        return c, pct

    fam_count, fam_pct = occupancy("family")
    pol_count, pol_pct = occupancy("polymorph")
    return {
        "model_assignments": pred,
        "family_counts": fam_count,
        "family_percent": fam_pct,
        "polymorph_counts": pol_count,
        "polymorph_percent": pol_pct,
    }


def plot_projection_heatmap(pct: pd.DataFrame, title: str, out_png: Path, out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(10, 0.55 * pct.shape[1] + 5), 7))
    im = ax.imshow(pct.to_numpy(), aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(np.arange(pct.shape[1])); ax.set_xticklabels(pct.columns, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(pct.shape[0])); ax.set_yticklabels(pct.index, fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("Prediction occupancy (%)")
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Experimental structural group")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ======================================================================================
# Metadata association tests
# ======================================================================================

def cramers_v(table: pd.DataFrame) -> float:
    if table.size == 0:
        return float("nan")
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return math.sqrt(chi2 / (n * k)) if n > 0 and k > 0 else float("nan")


def permutation_chi_square(df: pd.DataFrame, cluster_col: str, meta_col: str, n_perm: int, rng: np.random.Generator) -> dict:
    d = df[[cluster_col, meta_col]].dropna().copy()
    d = d[(d[meta_col].astype(str) != "")]
    if d[cluster_col].nunique() < 2 or d[meta_col].nunique() < 2:
        return {"n": len(d), "n_clusters": d[cluster_col].nunique(), "n_categories": d[meta_col].nunique(), "statistic": np.nan, "p_permutation": np.nan, "cramers_v": np.nan}
    table = pd.crosstab(d[cluster_col], d[meta_col])
    obs, _, _, _ = chi2_contingency(table, correction=False)
    meta = d[meta_col].to_numpy(copy=True)
    clusters = d[cluster_col].to_numpy()
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(meta)
        tab = pd.crosstab(pd.Series(clusters), pd.Series(perm))
        stat, _, _, _ = chi2_contingency(tab, correction=False)
        if stat >= obs - 1e-12:
            ge += 1
    return {
        "n": len(d), "n_clusters": d[cluster_col].nunique(), "n_categories": d[meta_col].nunique(),
        "statistic": obs, "p_permutation": (ge + 1) / (n_perm + 1), "cramers_v": cramers_v(table),
    }


def continuous_family_test(df: pd.DataFrame, cluster_col: str, value_col: str) -> dict:
    d = df[[cluster_col, value_col]].dropna().copy()
    groups = [g[value_col].to_numpy(float) for _, g in d.groupby(cluster_col) if len(g) >= 2]
    if len(groups) < 2:
        return {"n": len(d), "n_groups_with_n_ge_2": len(groups), "statistic": np.nan, "p_value": np.nan}
    stat, p = kruskal(*groups)
    return {"n": len(d), "n_groups_with_n_ge_2": len(groups), "statistic": stat, "p_value": p}


def run_metadata_associations(tax: pd.DataFrame, peptide_key: str, n_perm: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260905)
    rows = []
    for level in ("family", "polymorph"):
        specs = [
            ("source_class", "categorical", tax),
            ("method_group", "categorical", tax),
            ("modification_status", "categorical", tax),
            ("disease_group", "categorical", tax[(tax.source_class == "Human-derived") & (tax.disease_group != "Not reported / non-human")]),
            ("tissue_group", "categorical", tax[(tax.source_class == "Human-derived") & (tax.tissue_group != "Human tissue - unspecified")]),
            ("coordinate_fraction_of_reference", "continuous", tax),
            ("em_resolution_A", "continuous", tax[tax.method_group == "Electron microscopy"]),
        ]
        for variable, typ, subset in specs:
            if typ == "categorical":
                res = permutation_chi_square(subset, level, variable, n_perm, rng)
                rows.append({
                    "peptide": PEPTIDE_INFO[peptide_key]["label"], "hierarchy_level": level,
                    "metadata_variable": variable, "test": "permutation chi-square",
                    **res,
                })
            else:
                res = continuous_family_test(subset, level, variable)
                rows.append({
                    "peptide": PEPTIDE_INFO[peptide_key]["label"], "hierarchy_level": level,
                    "metadata_variable": variable, "test": "Kruskal-Wallis",
                    **res,
                })
    return pd.DataFrame(rows)


# ======================================================================================
# Preflight / orchestration
# ======================================================================================

def preflight(project: Path) -> None:
    banner("Aβ ECOD/CATH-INSPIRED HIERARCHY PREFLIGHT")
    required = [
        project / "experimental" / "metadata" / "abeta_metadata_clean.tsv",
        project / "experimental" / "metadata" / "AB40_Amyloid_Explorer.csv",
        project / "experimental" / "metadata" / "ABETA_42_Amyloid_Explorer.csv",
        project / "experimental" / "metadata" / "chain_summary.tsv",
        project / "experimental" / "metadata" / "representative_chain_decisions.tsv",
        project / "experimental" / "foldseek" / "matrices" / "qtmscore_matrix.csv",
        project / "experimental" / "foldseek" / "matrices" / "alntmscore_matrix.csv",
        project / "experimental" / "foldseek" / "matrices" / "rmsd_matrix.csv",
        project / "experimental" / "structures" / "representatives_pdb_79",
        project / "analysis" / "foldseek" / "ab40_separate" / "assignments" / "model_level_assignments_720.tsv",
        project / "analysis" / "foldseek" / "ab42_separate" / "assignments" / "model_level_assignments_720.tsv",
    ]
    missing = [p for p in required if not p.exists()]
    print(f"Project: {project}")
    for p in required:
        print(f"  {'OK' if p.exists() else 'MISSING'}  {p}")
    if missing:
        raise FileNotFoundError("Missing required inputs:\n  " + "\n  ".join(map(str, missing)))

    meta, report = build_enriched_metadata(project, write_outputs=False)
    print("\n" + report)
    for key, info in PEPTIDE_INFO.items():
        mats = get_experimental_matrices(project, meta, key)
        print(f"{info['label']} qTM subset: {mats['qtm'].shape[0]} × {mats['qtm'].shape[1]}")
    print("\nPREFLIGHT PASSED. No source files were modified.")


def build_final(project: Path, metadata: pd.DataFrame, cuts: dict[str, tuple[float, float]], n_perm: int) -> None:
    tax_out = project / "analysis" / "ecod_cath_hierarchy" / "taxonomy"
    proj_out = project / "analysis" / "ecod_cath_hierarchy" / "prediction_projection"
    assoc_out = project / "analysis" / "ecod_cath_hierarchy" / "associations"
    fig_tax = project / "figures" / "ecod_cath_hierarchy" / "taxonomy"
    fig_proj = project / "figures" / "ecod_cath_hierarchy" / "prediction_projection"
    for d in (tax_out, proj_out, assoc_out, fig_tax, fig_proj): d.mkdir(parents=True, exist_ok=True)

    all_assoc = []
    for key, info in PEPTIDE_INFO.items():
        family_cut, polymorph_cut = cuts[key]
        banner(f"BUILDING FINAL {info['label']} ECOD/CATH-INSPIRED TAXONOMY")
        mats = get_experimental_matrices(project, metadata, key)
        qtm = mats["qtm"]
        dist, Z, order = hierarchy_from_qtm(qtm)
        tax = build_taxonomy_assignments(metadata, key, qtm, Z, order, family_cut, polymorph_cut)
        tax.to_csv(tax_out / f"{key}_taxonomy_assignments.tsv", sep="\t", index=False)
        fam, pol = family_polymorph_summaries(tax)
        fam.to_csv(tax_out / f"{key}_family_summary.tsv", sep="\t", index=False)
        pol.to_csv(tax_out / f"{key}_polymorph_summary.tsv", sep="\t", index=False)

        ordered = tax["structure"].tolist()
        qtm_ord = qtm.loc[ordered, ordered]
        qtm_ord.to_csv(tax_out / f"{key}_qtm_same_taxonomy_order.csv", float_format="%.6f")

        # Contact Jaccard over shared modelled residue positions, derived directly from representative PDB coordinates.
        jac, common = compute_contact_jaccard_matrix(project, metadata, key, ordered)
        jac.to_csv(tax_out / f"{key}_contact_jaccard_shared_residues.csv", float_format="%.6f")
        common.to_csv(tax_out / f"{key}_contact_common_residue_fraction.csv", float_format="%.6f")

        if "rmsd" in mats:
            mats["rmsd"].loc[ordered, ordered].to_csv(tax_out / f"{key}_rmsd_same_taxonomy_order.csv", float_format="%.6f")
        if "alntm" in mats:
            mats["alntm"].loc[ordered, ordered].to_csv(tax_out / f"{key}_alntm_same_taxonomy_order.csv", float_format="%.6f")

        plot_integrated_taxonomy(
            tax, qtm, Z, order, key, family_cut, polymorph_cut,
            fig_tax / f"{key}_ecod_cath_integrated_qtm_metadata.png",
            fig_tax / f"{key}_ecod_cath_integrated_qtm_metadata.pdf",
        )
        plot_companion_matrix(
            jac, tax,
            f"{info['label']} contact-map agreement in ECOD/CATH-inspired taxonomy order\nJaccard calculated only across residue positions modelled in both structures",
            "Contact Jaccard", fig_tax / f"{key}_contact_jaccard_same_order.png", fig_tax / f"{key}_contact_jaccard_same_order.pdf",
            vmin=0, vmax=1,
        )
        if "rmsd" in mats:
            plot_companion_matrix(
                mats["rmsd"], tax,
                f"{info['label']} RMSD in ECOD/CATH-inspired taxonomy order",
                "RMSD (Å)", fig_tax / f"{key}_rmsd_same_order.png", fig_tax / f"{key}_rmsd_same_order.pdf",
                vmin=0, vmax=None, cmap="magma_r",
            )
        if "alntm" in mats:
            plot_companion_matrix(
                mats["alntm"], tax,
                f"{info['label']} alignment-TM in ECOD/CATH-inspired taxonomy order",
                "Symmetric alnTM", fig_tax / f"{key}_alntm_same_order.png", fig_tax / f"{key}_alntm_same_order.pdf",
                vmin=0, vmax=1,
            )

        proj = project_predictions(project, tax, key)
        for name, df in proj.items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(proj_out / f"{key}_{name}.tsv", sep="\t")
        plot_projection_heatmap(
            proj["family_percent"],
            f"{info['label']} prediction occupancy across experimentally defined structural families",
            fig_proj / f"{key}_prediction_family_occupancy.png", fig_proj / f"{key}_prediction_family_occupancy.pdf",
        )
        plot_projection_heatmap(
            proj["polymorph_percent"],
            f"{info['label']} prediction occupancy across experimentally defined polymorph groups",
            fig_proj / f"{key}_prediction_polymorph_occupancy.png", fig_proj / f"{key}_prediction_polymorph_occupancy.pdf",
        )

        assoc = run_metadata_associations(tax, key, n_perm)
        assoc.to_csv(assoc_out / f"{key}_metadata_associations.tsv", sep="\t", index=False)
        all_assoc.append(assoc)

        print(f"Family cut distance:    {family_cut:.4f} (qTM-equivalent similarity {1-family_cut:.4f})")
        print(f"Polymorph cut distance: {polymorph_cut:.4f} (qTM-equivalent similarity {1-polymorph_cut:.4f})")
        print(f"Families:   {tax['family'].nunique()}")
        print(f"Polymorphs: {tax['polymorph'].nunique()}")
        print(f"Prediction assignments projected: {len(proj['model_assignments'])}")

    pd.concat(all_assoc, ignore_index=True).to_csv(assoc_out / "ab40_ab42_metadata_associations.tsv", sep="\t", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    action = p.add_mutually_exclusive_group(required=False)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--metadata-only", action="store_true")
    action.add_argument("--diagnostics", action="store_true")
    action.add_argument("--build-taxonomy", action="store_true")
    action.add_argument("--all", action="store_true", help="Metadata + diagnostics; if all four cut arguments are supplied, also build final taxonomy.")
    p.add_argument("--ab40-family-cut", type=float)
    p.add_argument("--ab40-polymorph-cut", type=float)
    p.add_argument("--ab42-family-cut", type=float)
    p.add_argument("--ab42-polymorph-cut", type=float)
    p.add_argument("--permutations", type=int, default=5000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project.expanduser().resolve()

    if not any([args.preflight, args.metadata_only, args.diagnostics, args.build_taxonomy, args.all]):
        args.preflight = True

    if args.preflight:
        preflight(project)
        return 0

    banner("REPAIRING / ENRICHING EXPERIMENTAL METADATA")
    metadata, report = build_enriched_metadata(project, write_outputs=True)
    print(report)
    print("Derived metadata written under:")
    print(project / "analysis" / "ecod_cath_hierarchy" / "metadata")

    if args.metadata_only:
        return 0

    if args.diagnostics or args.all:
        run_diagnostics(project, metadata)
        print("\nHierarchy diagnostics complete. No final family/polymorph cut was imposed.")
        print("Review the candidate-cut TSVs and diagnostic figures before building the final taxonomy.")

    if args.build_taxonomy or args.all:
        vals = [args.ab40_family_cut, args.ab40_polymorph_cut, args.ab42_family_cut, args.ab42_polymorph_cut]
        if any(v is None for v in vals):
            if args.build_taxonomy:
                raise ValueError("--build-taxonomy requires all four peptide-specific family/polymorph cut arguments.")
            print("\nFinal taxonomy build skipped because explicit cut heights were not supplied.")
            return 0
        cuts = {
            "ab40": (float(args.ab40_family_cut), float(args.ab40_polymorph_cut)),
            "ab42": (float(args.ab42_family_cut), float(args.ab42_polymorph_cut)),
        }
        build_final(project, metadata, cuts, int(args.permutations))
        print("\nFINAL ECOD/CATH-INSPIRED TAXONOMY BUILD COMPLETE.")
        print("Analysis:", project / "analysis" / "ecod_cath_hierarchy")
        print("Figures: ", project / "figures" / "ecod_cath_hierarchy")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
