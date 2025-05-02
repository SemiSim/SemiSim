Up to date as of 24 APRIL 2025 11:02A -Ry

Parsers integrated so far:
_saps.txt *fixed
_pondr.txt
_plaac.tsv *fixed
_string.tsv *fixed 
_pdb.pdb *removed
_dssp.mmcif *removed
_biogrid.tsv (individual protein) *fixed

Each parser is implemented in its own function for clarity and extensibility:) makes it much easier to add more parsing functions on as I go. I can annotate it when i have the final code up so we all know exactly what's doing what and why

Need to install Biopython & MD Analysis (optional-ish fallback)
```bash
pip install biopython mdanalysis

```

Run script with:
```bash
python parser_v4.py
```

==parser_v4.py==
```python
import os
import re 
import pandas as pd 
import numpy as np 
from glob import glob
from itertools import groupby

def parse_pondr(filepath):
    scores = []
    with open(filepath, 'r') as f:
        for line in f:
            if re.match(r'^\d+\s+\w\s+[\d\.]+', line):
                parts = line.strip().split()
                scores.append(float(parts[2]))
    if scores:
        return {
            'pond_mean_disorder': np.mean(scores),
            'pond_max_disorder': np.max(scores),
            'pond_disordered_frac': sum(s > 0.5 for s in scores) / len(scores),
            'pond_longest_disordered': max([len(list(g)) for k, g in groupby(scores, lambda x: x > 0.5) if k] or [0])
        }
    return {}

def parse_saps(filepath):
   
    features = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()

    comp_re = re.compile(r'^([A-Z])\s*:\s*\d+\(\s*([\d\.]+)%\)')
    charge_re = re.compile(r'^(KRH|ED)\s*:\s*\d+\s*\(\s*([\d\.]+)%\)')
    hydro_re = re.compile(r'Hydrophobicity.*?mean\)\s*:\s*([-+]?\d*\.\d+)')

    for line in lines:
        m = comp_re.match(line)
        if m:
            aa, pct = m.groups()
            features[f'saps_aa_{aa}'] = float(pct) / 100.0
            continue

        m = charge_re.match(line)
        if m:
            group, pct = m.groups()
            key = 'saps_pct_' + ('positively_charged' if group=='KRH' else 'negatively_charged')
            features[key] = float(pct) / 100.0
            continue

        m = hydro_re.search(line)
        if m:
            features['saps_mean_hydropathy'] = float(m.group(1))
            continue

    return features

def parse_plaac(filepath):
    df = pd.read_csv(filepath, sep='\t', comment='#')

    plaac_mean = df['PLAAC'].mean()
    plaac_max  = df['PLAAC'].max()
    probs      = df['HMM.PrD-like']
    prd_mask   = probs > 0.5
    prd_frac   = prd_mask.mean()
    prd_mean_prob = probs.mean()
    prd_max_prob  = probs.max()
    prd_longest = max((len(list(g)) for k, g in groupby(prd_mask) if k),
                      default=0)
    regions = []
    start = None
    for i, val in enumerate(prd_mask):
        if val and start is None:
            start = i
        if not val and start is not None:
            regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(prd_mask)))

    if regions:
        longest = max(regions, key=lambda x: x[1] - x[0])
        region_df = df.iloc[longest[0]:longest[1]]
        charge_mean = region_df['CHARGE'].mean()
        hydro_mean  = region_df['HYDRO'].mean()
    else:
        charge_mean = np.nan
        hydro_mean  = np.nan

    return {
        'plaac_mean': plaac_mean,
        'plaac_max': plaac_max,
        'prd_frac': prd_frac,
        'prd_mean_prob': prd_mean_prob,
        'prd_max_prob': prd_max_prob,
        'prd_longest': prd_longest,
        'prd_region_charge_mean': charge_mean,
        'prd_region_hydro_mean': hydro_mean
    }

def parse_string(filepath):
    df = pd.read_csv(filepath, sep='\t')

    scores = pd.to_numeric(df['Score'], errors='coerce')

    return {
        'string_ppi_count': len(df),
        'string_mean_score': scores.mean(),
        'string_high_conf_interactors': (scores > 700).sum()
    }

def parse_biogrid(filepath):
    df = pd.read_csv(filepath, sep='\t', comment='#', low_memory=False)

    col_pairs = [
        ('Official Symbol Interactor A', 'Official Symbol Interactor B'),
        ('Systematic Name Interactor A', 'Systematic Name Interactor B')
    ]

    bait_col = None
    partner_col = None

    for a_col, b_col in col_pairs:
        if a_col in df.columns and df[a_col].nunique(dropna=True) == 1:
            bait_col = a_col
            partner_col = b_col
            break
        if b_col in df.columns and df[b_col].nunique(dropna=True) == 1:
            bait_col = b_col
            partner_col = a_col
            break

    if bait_col is None:
        raise ValueError(f"Could not auto-detect bait column in {filepath}")

    bait_symbol = df[bait_col].iloc[0]
    partners = set(df[partner_col].dropna().astype(str))
    
    scores = pd.to_numeric(df['Score'], errors='coerce')
     
    mods = df.get('Modification', pd.Series([], dtype=str)).dropna().astype(str)
)

    features = {
        'biogrid_bait': bait_symbol,
        'biogrid_num_partners': len(partners),
        'biogrid_partners_list': ';'.join(sorted(partners)),
        'biogrid_row_count': len(df),
        'biogrid_mean_score': scores.mean(),
        'biogrid_high_score_count': (scores > 700).sum()
    }

    for mod_type, count in mods.value_counts().items():
        key = f'biogrid_ptm_type_{mod_type}'
        features[key] = count

    return features

def build_feature_matrix(data_dir):
    proteins = [
        os.path.basename(f).split('_')[0]
        for f in glob(os.path.join(data_dir, 'pondr', '*.txt'))
    ]
    feature_rows = []

    for pid in proteins:
        row = {'protein_id': pid}

        try:
            row.update(parse_pondr(os.path.join(data_dir, 'pondr', f'{pid}_pondr.txt')))
        except Exception as e:
            print(f"[PONDR] {pid}:", e)

        try:
            row.update(parse_saps(os.path.join(data_dir, 'saps', f'{pid}_saps.txt')))
        except Exception as e:
            print(f"[SAPS] {pid}:, {e}")

        try:
            row.update(parse_plaac(os.path.join(data_dir, 'plaac', f'{pid}_plaac.txt')))
        except Exception as e:
            print(f"[PLAAC] {pid}: {e}")

        try:
            row.update(parse_string(os.path.join(data_dir, 'string', f'{pid}_string.txt')))
        except Exception as e:
            print(f"[STRING] {pid}: {e}")

        try:
            row.update(parse_pdb(os.path.join(data_dir, 'pdb', f'{pid}.pdb')))
        except Exception as e:
            print(f"[PDB] {pid}: {e}")

        try:
            row.update(parse_dssp(os.path.join(data_dir, 'dssp', f'{pid}.cif')))
        except Exception as e:
            print(f"[DSSP] {pid}: {e}")

        try:
            row.update(parse_biogrid(os.path.join(data_dir, 'biogrid', f'{pid}_biogrid.txt')))
        except Exception as e:
            print(f"[BIOGRID] {pid}: {e}")

        feature_rows.append(row)

    df = pd.DataFrame(feature_rows)
    df.set_index('protein_id', inplace=True)
    return df.fillna(0)

```























































