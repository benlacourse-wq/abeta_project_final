#!/usr/bin/env python3
"""
Final refinement pass for the completed Aβ40/Aβ42 ECOD/CATH-inspired taxonomy.

This script DOES NOT rebuild or change the structural taxonomy. It reads the final
V3 outputs and writes only new derived refinement outputs.

Refinements:
  1) improved integrated taxonomy figures with explicit metadata legends and a
     four-level experimental-source annotation (direct human, human-seeded,
     animal, in-vitro/synthetic/unclear);
  2) BH-FDR corrected metadata-association tables, replacing the old 3-level
     source test with the refined 4-level provenance test;
  3) paired template-switching analysis at fold-class / family / polymorph level;
  4) Aβ42 coordinate-coverage sensitivity analysis (taxonomy unchanged);
  5) Contact-Jaccard figures where undefined comparisons are explicitly labelled.

No source structures, Foldseek results, V3 taxonomy assignments, or original
metadata files are modified.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import chi2_contingency

PROJECT_DEFAULT = Path('/home/benla/abeta_project_final')
BASE_REL = Path('analysis/ecod_cath_hierarchy')
FIG_REL = Path('figures/ecod_cath_hierarchy')

CUTS = {
    'ab40': {'class': 0.6096, 'family': 0.5436, 'polymorph': 0.4855},
    'ab42': {'class': 0.6071, 'family': 0.5127, 'polymorph': 0.3744},
}
LABELS = {'ab40': 'Aβ40', 'ab42': 'Aβ42'}
EXPECTED = {'ab40': 33, 'ab42': 46}

SOURCE_PALETTE = {
    'Human direct / ex vivo': '#1b9e77',
    'Human-derived seeded preparation': '#66c2a5',
    'Animal-derived': '#d95f02',
    'In vitro / synthetic / unclear': '#7570b3',
}
METHOD_PALETTE = {
    'Electron microscopy': '#4c78a8',
    'Solid-state NMR': '#f58518',
    'Solution NMR': '#e45756',
}
MOD_PALETTE = {'WT / unmodified': '#bdbdbd', 'Modified / mutant': '#e7298a'}
DISEASE_PALETTE = {
    'Not reported / non-human': '#eeeeee', 'AD': '#e41a1c', 'CAA': '#377eb8',
    'AD + CAA': '#984ea3', 'Down syndrome': '#4daf4a', 'Other neurological': '#ff7f00',
}
TISSUE_PALETTE = {
    'In vitro / not applicable': '#eeeeee', 'Meningeal / vascular': '#1f78b4',
    'Cortical / parenchymal brain': '#33a02c', 'Animal model': '#e31a1c',
    'Human tissue - unspecified': '#6a3d9a',
}


def banner(s: str) -> None:
    print('\n' + '=' * 100)
    print(s)
    print('=' * 100)


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def load_inputs(project: Path, pep: str):
    base = project / BASE_REL
    taxp = base / 'taxonomy' / f'{pep}_taxonomy_assignments.tsv'
    qtmp = base / 'taxonomy' / f'{pep}_qtm_same_taxonomy_order.csv'
    jacp = base / 'taxonomy' / f'{pep}_contact_jaccard_shared_residues.csv'
    assocp = base / 'associations' / f'{pep}_metadata_associations.tsv'
    predp = base / 'prediction_projection' / f'{pep}_model_assignments.tsv'
    for p in (taxp, qtmp, jacp, assocp, predp): require(p)
    tax = pd.read_csv(taxp, sep='\t').sort_values('dendrogram_position').reset_index(drop=True)
    qtm = pd.read_csv(qtmp, index_col=0)
    jac = pd.read_csv(jacp, index_col=0)
    assoc = pd.read_csv(assocp, sep='\t')
    pred = pd.read_csv(predp, sep='\t')
    if len(tax) != EXPECTED[pep]:
        raise ValueError(f'{pep}: expected {EXPECTED[pep]} taxonomy rows, found {len(tax)}')
    return tax, qtm, jac, assoc, pred


def refined_source(row: pd.Series) -> str:
    sc = str(row.get('source_class', ''))
    sub = str(row.get('source_subtype', ''))
    if sc == 'Human-derived':
        if 'seed' in sub.lower():
            return 'Human-derived seeded preparation'
        return 'Human direct / ex vivo'
    if sc == 'Animal-derived':
        return 'Animal-derived'
    return 'In vitro / synthetic / unclear'


def add_refined_source(tax: pd.DataFrame) -> pd.DataFrame:
    out = tax.copy()
    out['source_provenance_refined'] = out.apply(refined_source, axis=1)
    return out


def make_palette(values, cmap_name='tab20'):
    vals = list(dict.fromkeys(map(str, values)))
    cmap = plt.get_cmap(cmap_name, max(1, len(vals)))
    return {v: cmap(i) for i, v in enumerate(vals)}


def draw_strip(ax, values, palette, label):
    vals = [str(v) for v in values]
    colors = [palette.get(v, '#ffffff') for v in vals]
    arr = np.arange(len(vals))[None, :]
    ax.imshow(arr, aspect='auto', interpolation='nearest', cmap=ListedColormap(colors),
              vmin=0, vmax=max(1, len(vals)-1))
    ax.set_yticks([0]); ax.set_yticklabels([label], fontsize=8)
    ax.set_xticks([])
    for s in ax.spines.values(): s.set_visible(False)


def boundaries(vals):
    vals = list(vals)
    return [i for i in range(1, len(vals)) if vals[i] != vals[i-1]]


def linkage_from_qtm(qtm: pd.DataFrame, ids: list[str]) -> np.ndarray:
    m = qtm.loc[ids, ids].to_numpy(float)
    m = (m + m.T) / 2.0
    np.fill_diagonal(m, 1.0)
    d = np.clip(1.0 - m, 0.0, 1.0)
    np.fill_diagonal(d, 0.0)
    return linkage(squareform(d, checks=False), method='average')


def legend_block(ax, title, palette, values=None, fontsize=8):
    ax.axis('off')
    vals = list(palette.keys()) if values is None else [v for v in palette if v in set(map(str, values))]
    handles = [Patch(facecolor=palette[v], edgecolor='none', label=v) for v in vals]
    if handles:
        ax.legend(handles=handles, title=title, loc='upper left', frameon=False,
                  fontsize=fontsize, title_fontsize=fontsize+1, borderaxespad=0)


def plot_integrated_refined(tax, qtm, pep, outpng, outpdf):
    label = LABELS[pep]
    ids = tax['structure'].tolist()
    n = len(ids)
    mat = qtm.loc[ids, ids].to_numpy(float)
    Z = linkage_from_qtm(qtm, ids)

    class_pal = make_palette(tax['fold_class'].unique(), 'Set2')
    fam_pal = make_palette(tax['family'].unique(), 'tab20')
    pol_pal = make_palette(tax['polymorph'].unique(), 'gist_ncar')

    fig = plt.figure(figsize=(22, 19 if n <= 35 else 21))
    gs = gridspec.GridSpec(
        11, 4,
        height_ratios=[3.0, .23,.23,.23,.23,.23,.23,.23,.23,.23,10.0],
        width_ratios=[1.0, 10.0, .55, 3.6],
        hspace=.05, wspace=.10,
    )

    axd = fig.add_subplot(gs[0,1])
    dendrogram(Z, no_labels=True, color_threshold=0, above_threshold_color='black', ax=axd)
    axd.axhline(CUTS[pep]['class'], ls='-.', lw=1.2, label=f"Fold-class cut {CUTS[pep]['class']:.4f}")
    axd.axhline(CUTS[pep]['family'], ls='--', lw=1.2, label=f"Family cut {CUTS[pep]['family']:.4f}")
    axd.axhline(CUTS[pep]['polymorph'], ls=':', lw=1.2, label=f"Polymorph cut {CUTS[pep]['polymorph']:.4f}")
    axd.set_ylabel('1 − symmetric qTM'); axd.set_xticks([])
    axd.legend(frameon=False, fontsize=8, loc='upper right')

    tracks = [
        ('fold_class', class_pal, 'Fold class'),
        ('family', fam_pal, 'Family'),
        ('polymorph', pol_pal, 'Polymorph'),
        ('source_provenance_refined', SOURCE_PALETTE, 'Source'),
        ('method_group', METHOD_PALETTE, 'Method'),
        ('modification_status', MOD_PALETTE, 'Modified'),
        ('disease_group', DISEASE_PALETTE, 'Disease'),
        ('tissue_group', TISSUE_PALETTE, 'Tissue'),
    ]
    for r,(col,pal,lab) in enumerate(tracks, start=1):
        draw_strip(fig.add_subplot(gs[r,1]), tax[col].astype(str).tolist(), pal, lab)

    axc = fig.add_subplot(gs[9,1])
    cov = tax['coordinate_fraction_of_reference'].to_numpy(float)[None,:]
    covim = axc.imshow(cov, aspect='auto', interpolation='nearest', cmap='viridis', vmin=0, vmax=1)
    axc.set_yticks([0]); axc.set_yticklabels(['Coord. fraction'], fontsize=8); axc.set_xticks([])
    for s in axc.spines.values(): s.set_visible(False)

    ax = fig.add_subplot(gs[10,1])
    im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=1, interpolation='nearest', aspect='equal')
    pdbs = tax['pdb_id'].astype(str).tolist()
    ax.set_xticks(np.arange(n)); ax.set_xticklabels(pdbs, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(n)); ax.set_yticklabels(pdbs, fontsize=7)
    ax.set_xlabel('Experimental representative'); ax.set_ylabel('Experimental representative')
    for b in boundaries(tax['polymorph']):
        ax.axhline(b-.5,c='white',lw=.45,alpha=.85); ax.axvline(b-.5,c='white',lw=.45,alpha=.85)
    for b in boundaries(tax['family']):
        ax.axhline(b-.5,c='white',lw=1.4); ax.axvline(b-.5,c='white',lw=1.4)
    for b in boundaries(tax['fold_class']):
        ax.axhline(b-.5,c='white',lw=2.6); ax.axvline(b-.5,c='white',lw=2.6)

    cax = fig.add_subplot(gs[10,2]); cb=fig.colorbar(im,cax=cax); cb.set_label('Symmetric qTM')
    covax = fig.add_subplot(gs[9,2]); cbc=fig.colorbar(covim,cax=covax,orientation='horizontal'); cbc.set_label('Coordinate fraction', fontsize=8); cbc.ax.tick_params(labelsize=7)

    # Right-side metadata legends. Family/polymorph colours are hierarchy tracks, not metadata legends.
    subgs = gridspec.GridSpecFromSubplotSpec(6,1,subplot_spec=gs[:,3],hspace=.45)
    legend_block(fig.add_subplot(subgs[0]), 'Fold class', class_pal, tax['fold_class'])
    legend_block(fig.add_subplot(subgs[1]), 'Experimental source', SOURCE_PALETTE, tax['source_provenance_refined'])
    legend_block(fig.add_subplot(subgs[2]), 'Experimental method', METHOD_PALETTE, tax['method_group'])
    legend_block(fig.add_subplot(subgs[3]), 'Modification status', MOD_PALETTE, tax['modification_status'])
    legend_block(fig.add_subplot(subgs[4]), 'Disease annotation', DISEASE_PALETTE, tax['disease_group'])
    legend_block(fig.add_subplot(subgs[5]), 'Tissue/source annotation', TISSUE_PALETTE, tax['tissue_group'])

    fig.suptitle(
        f'{label} ECOD/CATH-inspired experimental structural taxonomy\n'
        'Experimental structures define the hierarchy; metadata are annotations only',
        fontsize=17, weight='bold', y=.995,
    )
    fig.savefig(outpng,dpi=300,bbox_inches='tight'); fig.savefig(outpdf,bbox_inches='tight'); plt.close(fig)


def permutation_chi_square(df, cluster_col, meta_col, n_perm=5000, seed=20260906):
    d = df[[cluster_col, meta_col]].dropna().copy()
    d = d[d[meta_col].astype(str) != '']
    if d[cluster_col].nunique() < 2 or d[meta_col].nunique() < 2:
        return dict(n=len(d), n_clusters=d[cluster_col].nunique(), n_categories=d[meta_col].nunique(), statistic=np.nan, p_permutation=np.nan, cramers_v=np.nan)
    table = pd.crosstab(d[cluster_col], d[meta_col])
    obs,_,_,_ = chi2_contingency(table, correction=False)
    n = table.values.sum(); k=min(table.shape)-1
    v = math.sqrt(obs/(n*k)) if n>0 and k>0 else np.nan
    rng=np.random.default_rng(seed)
    meta=d[meta_col].to_numpy(copy=True); clusters=d[cluster_col].to_numpy(copy=True)
    ge=0
    for _ in range(n_perm):
        tab=pd.crosstab(pd.Series(clusters), pd.Series(rng.permutation(meta)))
        stat,_,_,_=chi2_contingency(tab, correction=False)
        if stat >= obs - 1e-12: ge += 1
    return dict(n=len(d), n_clusters=d[cluster_col].nunique(), n_categories=d[meta_col].nunique(), statistic=obs, p_permutation=(ge+1)/(n_perm+1), cramers_v=v)


def bh_fdr(p):
    arr=np.asarray(p,float); out=np.full(len(arr),np.nan)
    ok=np.isfinite(arr); vals=arr[ok]
    if len(vals)==0: return out
    order=np.argsort(vals); ranked=vals[order]
    q=ranked*len(vals)/np.arange(1,len(vals)+1)
    q=np.minimum.accumulate(q[::-1])[::-1]; q=np.clip(q,0,1)
    temp=np.empty(len(vals)); temp[order]=q; out[np.where(ok)[0]]=temp
    return out


def build_fdr_table(tax, assoc, pep, n_perm):
    a=assoc.copy()
    a=a[a['metadata_variable']!='source_class'].copy()
    rows=[]
    for level in ('fold_class','family','polymorph'):
        res=permutation_chi_square(tax,level,'source_provenance_refined',n_perm=n_perm,seed=20260906+len(rows))
        rows.append({'peptide':LABELS[pep],'hierarchy_level':level,'metadata_variable':'source_provenance_refined','test':'permutation chi-square',**res,'n_groups_with_n_ge_2':np.nan,'p_value':np.nan})
    a=pd.concat([a,pd.DataFrame(rows)],ignore_index=True,sort=False)
    a['p_raw_for_fdr']=a['p_permutation'].where(a['p_permutation'].notna(), a['p_value'])
    a['p_bh_fdr']=bh_fdr(a['p_raw_for_fdr'].to_numpy(float))
    a['bh_fdr_significant_0.05']=a['p_bh_fdr'] < .05
    order={'fold_class':0,'family':1,'polymorph':2}
    a['_o']=a['hierarchy_level'].map(order); a=a.sort_values(['_o','metadata_variable']).drop(columns='_o').reset_index(drop=True)
    return a


def plot_fdr_heatmap(df, pep, outpng, outpdf):
    vars_order=['source_provenance_refined','method_group','modification_status','disease_group','tissue_group','coordinate_fraction_of_reference','em_resolution_A']
    pretty={'source_provenance_refined':'Source provenance','method_group':'Experimental method','modification_status':'Modification status','disease_group':'Disease (human subset)','tissue_group':'Tissue/source (human subset)','coordinate_fraction_of_reference':'Coordinate fraction','em_resolution_A':'EM resolution'}
    levels=['fold_class','family','polymorph']; level_names=['Fold class','Family','Polymorph']
    m=np.full((len(vars_order),3),np.nan); annot=np.empty((len(vars_order),3),object)
    for i,v in enumerate(vars_order):
        for j,l in enumerate(levels):
            x=df[(df.metadata_variable==v)&(df.hierarchy_level==l)]
            if len(x):
                q=float(x.iloc[0].p_bh_fdr) if pd.notna(x.iloc[0].p_bh_fdr) else np.nan
                m[i,j]=-math.log10(max(q,1e-6)) if np.isfinite(q) else np.nan
                annot[i,j]=(f'q={q:.3g}' + (' *' if q<.05 else '')) if np.isfinite(q) else 'NA'
            else: annot[i,j]='NA'
    fig,ax=plt.subplots(figsize=(8.5,7))
    masked=np.ma.masked_invalid(m); im=ax.imshow(masked,cmap='viridis',vmin=0,vmax=max(2.0,np.nanmax(m)))
    ax.set_xticks(range(3)); ax.set_xticklabels(level_names)
    ax.set_yticks(range(len(vars_order))); ax.set_yticklabels([pretty[v] for v in vars_order])
    for i in range(len(vars_order)):
        for j in range(3): ax.text(j,i,annot[i,j],ha='center',va='center',fontsize=8,color='white' if np.isfinite(m[i,j]) and m[i,j]>1 else 'black')
    cb=fig.colorbar(im,ax=ax,pad=.02); cb.set_label('−log10(BH-FDR q)')
    ax.set_title(f'{LABELS[pep]} metadata associations with structural hierarchy\nBH-FDR corrected within peptide; * q < 0.05',weight='bold')
    fig.tight_layout(); fig.savefig(outpng,dpi=300,bbox_inches='tight'); fig.savefig(outpdf,bbox_inches='tight'); plt.close(fig)


def pair_template_predictions(pred, pep):
    key=['method','oligomer_size','model_rank']
    no=pred[pred.template_status.eq('no_templates')].copy()
    yes=pred[pred.template_status.eq('with_templates')].copy()
    cols=['foldseek_id','qtmscore','best_pdb_id','fold_class','family','polymorph']
    a=no[key+cols].rename(columns={c:f'{c}_no_templates' for c in cols})
    b=yes[key+cols].rename(columns={c:f'{c}_with_templates' for c in cols})
    p=a.merge(b,on=key,how='inner',validate='one_to_one')
    if len(p)!=360: raise ValueError(f'{pep}: expected 360 paired template comparisons, got {len(p)}')
    p['peptide']=LABELS[pep]
    for level in ('fold_class','family','polymorph'):
        p[f'{level}_switched']=p[f'{level}_no_templates'] != p[f'{level}_with_templates']
    p['delta_best_qtm']=p['qtmscore_with_templates']-p['qtmscore_no_templates']
    return p


def template_switch_summary(p):
    rows=[]
    display={'AlphaFold':'AlphaFold 3','Boltz':'Boltz-2','OpenFold':'OpenFold3','Protenix':'Protenix-v2'}
    for method,g in p.groupby('method',sort=False):
        for level in ('fold_class','family','polymorph'):
            sw=g[f'{level}_switched']
            rows.append({'peptide':g['peptide'].iloc[0],'method':method,'display_method':display.get(method,method),'hierarchy_level':level,'n_pairs':len(g),'n_switched':int(sw.sum()),'percent_switched':100*sw.mean(),'mean_delta_best_qtm':g.delta_best_qtm.mean(),'median_delta_best_qtm':g.delta_best_qtm.median()})
    return pd.DataFrame(rows)


def transition_table(p):
    rows=[]
    for method,g in p.groupby('method',sort=False):
        for level in ('fold_class','family','polymorph'):
            a=f'{level}_no_templates'; b=f'{level}_with_templates'
            t=g.groupby([a,b]).size().reset_index(name='n_pairs').sort_values('n_pairs',ascending=False)
            for _,r in t.iterrows():
                rows.append({'peptide':g.peptide.iloc[0],'method':method,'hierarchy_level':level,'no_templates_group':r[a],'with_templates_group':r[b],'n_pairs':int(r.n_pairs),'percent_of_method_pairs':100*int(r.n_pairs)/len(g)})
    return pd.DataFrame(rows)


def plot_template_switch(s, pep, outpng, outpdf):
    methods=['AlphaFold 3','Boltz-2','OpenFold3','Protenix-v2']; levels=['fold_class','family','polymorph']; labs=['Fold class','Family','Polymorph']
    m=np.zeros((4,3))
    for i,method in enumerate(methods):
        for j,l in enumerate(levels):
            x=s[(s.display_method==method)&(s.hierarchy_level==l)]
            m[i,j]=float(x.iloc[0].percent_switched)
    fig,ax=plt.subplots(figsize=(8,5.5)); im=ax.imshow(m,cmap='viridis',vmin=0,vmax=100,aspect='auto')
    ax.set_xticks(range(3)); ax.set_xticklabels(labs); ax.set_yticks(range(4)); ax.set_yticklabels(methods)
    for i in range(4):
        for j in range(3): ax.text(j,i,f'{m[i,j]:.1f}%',ha='center',va='center',color='white' if m[i,j]>50 else 'black',fontsize=10)
    cb=fig.colorbar(im,ax=ax,pad=.02); cb.set_label('Paired models switching structural group (%)')
    ax.set_title(f'{LABELS[pep]} template-associated switching across the experimental structural taxonomy\nPairs matched by method, oligomer size and model rank',weight='bold')
    fig.tight_layout(); fig.savefig(outpng,dpi=300,bbox_inches='tight'); fig.savefig(outpdf,bbox_inches='tight'); plt.close(fig)


def same_cluster_pair_metrics(full_labels, sens_labels):
    n=len(full_labels); agree=0; total=0; inter=0; union=0; full_same=0; sens_same=0
    for i in range(n):
        for j in range(i+1,n):
            a=full_labels[i]==full_labels[j]; b=sens_labels[i]==sens_labels[j]
            total+=1; agree += int(a==b); full_same += int(a); sens_same += int(b)
            inter += int(a and b); union += int(a or b)
    return {'pairwise_rand_agreement': agree/total if total else np.nan,
            'same_cluster_pair_jaccard': inter/union if union else np.nan,
            'n_same_pairs_full':full_same,'n_same_pairs_sensitivity':sens_same}


def coverage_sensitivity_ab42(tax, qtm, outdir_analysis, outdir_fig):
    thresholds=[0.60,0.70,0.75]
    rows=[]
    for threshold in thresholds:
        keep=tax[tax.coordinate_fraction_of_reference >= threshold].copy()
        ids=keep.structure.tolist(); subq=qtm.loc[ids,ids]
        Z=linkage_from_qtm(subq,ids)
        entry={'coverage_threshold':threshold,'n_retained':len(keep),'n_excluded':len(tax)-len(keep),'excluded_pdb_ids':'; '.join(tax.loc[tax.coordinate_fraction_of_reference<threshold,'pdb_id'].astype(str))}
        for level,cutname in [('fold_class','class'),('family','family'),('polymorph','polymorph')]:
            sens=fcluster(Z,t=CUTS['ab42'][cutname],criterion='distance')
            entry[f'{level}_n_groups']=len(set(sens)); entry[f'{level}_n_singletons']=int(pd.Series(sens).value_counts().eq(1).sum())
            met=same_cluster_pair_metrics(keep[level].astype(str).to_numpy(),sens)
            for k,v in met.items(): entry[f'{level}_{k}']=v
        rows.append(entry)
    summary=pd.DataFrame(rows)
    summary.to_csv(outdir_analysis/'ab42_coordinate_coverage_sensitivity_summary.tsv',sep='\t',index=False)

    # Same-cluster-pair Jaccard is deliberately used for the figure because it is
    # more stringent than overall Rand agreement (which can be dominated by the
    # many pairs that are different-cluster in both solutions).
    fig,ax=plt.subplots(figsize=(8,5.5))
    for level,label in [('fold_class','Fold class'),('family','Family'),('polymorph','Polymorph')]:
        ax.plot(summary.coverage_threshold*100,summary[f'{level}_same_cluster_pair_jaccard']*100,marker='o',label=label)
    for x,nret in zip(summary.coverage_threshold*100, summary.n_retained):
        ax.text(x,3,f'n={int(nret)}',ha='center',va='bottom',fontsize=8)
    ax.set_ylim(0,102); ax.set_xlabel('Minimum experimental coordinate coverage retained (%)'); ax.set_ylabel('Same-cluster pair Jaccard vs full taxonomy (%)')
    ax.set_title('Aβ42 coordinate-coverage sensitivity of the experimental taxonomy',weight='bold'); ax.legend(frameon=False); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(outdir_fig/'ab42_coordinate_coverage_sensitivity.png',dpi=300,bbox_inches='tight'); fig.savefig(outdir_fig/'ab42_coordinate_coverage_sensitivity.pdf',bbox_inches='tight'); plt.close(fig)

    # 70%-coverage dendrogram as the primary sensitivity visual.
    threshold=.70; keep=tax[tax.coordinate_fraction_of_reference>=threshold].copy(); ids=keep.structure.tolist(); Z=linkage_from_qtm(qtm.loc[ids,ids],ids)
    fig,ax=plt.subplots(figsize=(15,6))
    dendrogram(Z,labels=keep.set_index('structure').loc[ids,'pdb_id'].astype(str).tolist(),leaf_rotation=90,leaf_font_size=8,color_threshold=0,above_threshold_color='black',ax=ax)
    ax.axhline(CUTS['ab42']['class'],ls='-.',lw=1.2,label='Full-taxonomy fold-class cut')
    ax.axhline(CUTS['ab42']['family'],ls='--',lw=1.2,label='Full-taxonomy family cut')
    ax.axhline(CUTS['ab42']['polymorph'],ls=':',lw=1.2,label='Full-taxonomy polymorph cut')
    ax.set_ylabel('1 − symmetric qTM'); ax.set_title(f'Aβ42 sensitivity dendrogram after retaining structures with ≥70% coordinate coverage (n={len(keep)})',weight='bold'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(outdir_fig/'ab42_coordinate_coverage_sensitivity_dendrogram_70pct.png',dpi=300,bbox_inches='tight'); fig.savefig(outdir_fig/'ab42_coordinate_coverage_sensitivity_dendrogram_70pct.pdf',bbox_inches='tight'); plt.close(fig)
    return summary


def plot_contact_explicit(jac, tax, pep, outpng, outpdf):
    ids=tax.structure.tolist(); m=jac.loc[ids,ids].to_numpy(float); n=len(ids)
    masked=np.ma.masked_invalid(m)
    cmap=plt.get_cmap('viridis').copy(); cmap.set_bad('#d9d9d9')
    fig,ax=plt.subplots(figsize=(14.5,12))
    im=ax.imshow(masked,cmap=cmap,vmin=0,vmax=1,interpolation='nearest',aspect='equal')
    pdbs=tax.pdb_id.astype(str).tolist(); ax.set_xticks(np.arange(n)); ax.set_xticklabels(pdbs,rotation=90,fontsize=7); ax.set_yticks(np.arange(n)); ax.set_yticklabels(pdbs,fontsize=7)
    for b in boundaries(tax.polymorph): ax.axhline(b-.5,c='white',lw=.4,alpha=.8); ax.axvline(b-.5,c='white',lw=.4,alpha=.8)
    for b in boundaries(tax.family): ax.axhline(b-.5,c='white',lw=1.3); ax.axvline(b-.5,c='white',lw=1.3)
    for b in boundaries(tax.fold_class): ax.axhline(b-.5,c='white',lw=2.4); ax.axvline(b-.5,c='white',lw=2.4)
    cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.03); cb.set_label('Contact Jaccard')
    nundef=int(np.isnan(m).sum()); unique_undef=int((np.isnan(m).sum()-np.isnan(np.diag(m)).sum())/2)
    ax.legend(handles=[Patch(facecolor='#d9d9d9',edgecolor='black',label='Undefined: insufficient eligible common residue contacts')],loc='upper left',bbox_to_anchor=(0,-.13),frameon=False,fontsize=9)
    ax.set_title(f'{LABELS[pep]} Contact Jaccard in experimental taxonomy order\nGrey cells are undefined, not zero (unique undefined pairs: {unique_undef})',weight='bold')
    fig.tight_layout(); fig.savefig(outpng,dpi=300,bbox_inches='tight'); fig.savefig(outpdf,bbox_inches='tight'); plt.close(fig)


def preflight(project):
    banner('ECOD/CATH FINAL REFINEMENT PREFLIGHT')
    print('Project:',project)
    for pep in ('ab40','ab42'):
        tax,qtm,jac,assoc,pred=load_inputs(project,pep)
        rt=add_refined_source(tax)
        print(f"{LABELS[pep]}: taxonomy={len(tax)}, qTM={qtm.shape}, Contact Jaccard={jac.shape}, predictions={len(pred)}, associations={len(assoc)}")
        print('  refined source:', '; '.join(f'{k}={v}' for k,v in rt.source_provenance_refined.value_counts().items()))
        print('  undefined Contact-Jaccard matrix cells:', int(jac.isna().to_numpy().sum()))
    print('\nPREFLIGHT PASSED. V3 taxonomy will not be modified.')


def run_all(project, n_perm):
    aout=project/BASE_REL/'refinement_v4'; fout=project/FIG_REL/'refinement_v4'
    for d in (aout,fout): d.mkdir(parents=True,exist_ok=True)
    all_fdr=[]; all_switch=[]
    for pep in ('ab40','ab42'):
        banner(f'{LABELS[pep]} FINAL REFINEMENT')
        tax,qtm,jac,assoc,pred=load_inputs(project,pep); tax=add_refined_source(tax)
        # Refined derived taxonomy/metadata copy.
        tax.to_csv(aout/f'{pep}_taxonomy_assignments_refined_metadata.tsv',sep='\t',index=False)
        plot_integrated_refined(tax,qtm,pep,fout/f'{pep}_ecod_cath_integrated_qtm_metadata_refined.png',fout/f'{pep}_ecod_cath_integrated_qtm_metadata_refined.pdf')
        print('  refined integrated taxonomy figure: done')

        fdr=build_fdr_table(tax,assoc,pep,n_perm); fdr.to_csv(aout/f'{pep}_metadata_associations_bh_fdr.tsv',sep='\t',index=False); all_fdr.append(fdr)
        plot_fdr_heatmap(fdr,pep,fout/f'{pep}_metadata_associations_bh_fdr.png',fout/f'{pep}_metadata_associations_bh_fdr.pdf')
        print('  refined-source + BH-FDR association analysis: done')

        pairs=pair_template_predictions(pred,pep); pairs.to_csv(aout/f'{pep}_paired_template_switching_360.tsv',sep='\t',index=False)
        summ=template_switch_summary(pairs); summ.to_csv(aout/f'{pep}_template_switching_hierarchy_summary.tsv',sep='\t',index=False); all_switch.append(summ)
        trans=transition_table(pairs); trans.to_csv(aout/f'{pep}_template_switching_transitions.tsv',sep='\t',index=False)
        plot_template_switch(summ,pep,fout/f'{pep}_template_switching_class_family_polymorph.png',fout/f'{pep}_template_switching_class_family_polymorph.pdf')
        print('  paired template switching at class/family/polymorph: done')

        plot_contact_explicit(jac,tax,pep,fout/f'{pep}_contact_jaccard_undefined_explicit.png',fout/f'{pep}_contact_jaccard_undefined_explicit.pdf')
        print('  Contact Jaccard undefined-cell figure: done')

        if pep=='ab42':
            sens=coverage_sensitivity_ab42(tax,qtm,aout,fout)
            print('  Aβ42 coordinate-coverage sensitivity: done')
            print(sens[['coverage_threshold','n_retained','n_excluded','fold_class_pairwise_rand_agreement','family_pairwise_rand_agreement','polymorph_pairwise_rand_agreement']].to_string(index=False))

    pd.concat(all_fdr,ignore_index=True).to_csv(aout/'ab40_ab42_metadata_associations_bh_fdr.tsv',sep='\t',index=False)
    pd.concat(all_switch,ignore_index=True).to_csv(aout/'ab40_ab42_template_switching_hierarchy_summary.tsv',sep='\t',index=False)
    print('\nFINAL REFINEMENT COMPLETE')
    print('Analysis:',aout)
    print('Figures: ',fout)
    print('Original V3 taxonomy and all source data remain unchanged.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,default=PROJECT_DEFAULT)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--all',action='store_true')
    p.add_argument('--permutations',type=int,default=5000)
    args=p.parse_args()
    if not args.preflight and not args.all: args.preflight=True
    if args.preflight: preflight(args.project)
    if args.all: run_all(args.project,int(args.permutations))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
