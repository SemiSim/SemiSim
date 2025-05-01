
```bash
pip install pandas numpy scikit-learn pyyaml
```


`generate_yaml_config_v4.py`
```python
import os
import yaml

REQUIRED_DATA_SUBFOLDERS = ['pondr', 'plaac', 'saps', 'biogrid', 'string']

def ensure_dir(path):
    if not os.path.exists(path):
        print(f"Creating missing directory: {path}")
        os.makedirs(path, exist_ok=True)

def scan_project_structure(project_root='.'):
    config = {}

    data_dir = os.path.join(project_root, 'data')
    ensure_dir(data_dir)
    config['data_dir'] = 'data'

    for subfolder in REQUIRED_DATA_SUBFOLDERS:
        subfolder_path = os.path.join(data_dir, subfolder)
        ensure_dir(subfolder_path)

    llpsdb_path = os.path.join(data_dir, 'llpsdb.csv')
    print(f"Checking for llpsdb.csv at: {os.path.abspath(llpsdb_path)}")
    if os.path.isfile(llpsdb_path):
        config['llpsdb_csv'] = os.path.join('data', 'llpsdb.csv')
    else:
        print(f"llpsdb.csv not found at top level of {data_dir}. Searching recursively...")
        found = False
        for root, dirs, files in os.walk(data_dir):
        	print(f"Scanning {root} with files: {files}")
        	for fname in files:
        		if fname.strip().lower() == 'llpsdb.csv':
        			rel_path = os.path.relpath(os.path.join(root, fname)).replace("\\", "/")
        			config['llpsdb_csv'] = rel_path
        			print(f"Found llpsdb.csv at: {config['llpsdb_csv']}")
        			found = True
        			break
        	if found:
        		break
        if not found:
        	print("WARNING: llpsdb.csv not found anywhere in data/. Using fallback path.")
        	config['llpsdb_csv'] = os.path.join('data', 'llpsdb.csv')
        		
    fasta_dir = os.path.join(project_root, 'fasta')
    ensure_dir(fasta_dir)
    config['fasta_dir'] = 'fasta'

    config['output_csv'] = 'features.csv'
    config['missing_log'] = 'missing_files.log'
    config['origin_log'] = 'feature_origins.log'

    split_output_dir = os.path.join(project_root, 'split_data')
    ensure_dir(split_output_dir)
    config['split_output_dir'] = 'split_data'
    config['label_column'] = 'llps_label'
    config['test_size'] = 0.2
    config['random_state'] = 42

    return config

def write_config_yaml(config, output_path='config.yaml'):
    with open(output_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"\nGenerated config.yaml:")
    for k, v in config.items():
        print(f"  {k}: {v}")

if __name__ == '__main__':
    config = scan_project_structure()
    write_config_yaml(config)

```



`semisim_matrix.py`
```python
import os
import re
import yaml
import pandas as pd
import numpy as np
from glob import glob
from itertools import groupby
from sklearn.model_selection import train_test_split
from parse_v4.py import build_feature_matrix
from predict_secondary_structure_netsurfp_with_accessibility import predict_netsurfp_features

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_DIR = config.get("data_dir", "data")
FASTA_DIR = config.get("fasta_dir", "fasta")
LLPSDB_CSV = config.get("llpsdb_csv", "llpsdb.csv")
OUTPUT_CSV = config.get("output_csv", "features.csv")
MISSING_LOG = config.get("missing_log", "missing_files.log")
ORIGIN_LOG = config.get("origin_log", "feature_origins.log")

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
            key = 'saps_pct_' + ('positively_charged' if group == 'KRH' else 'negatively_charged')
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
    plaac_max = df['PLAAC'].max()
    probs = df['HMM.PrD-like']
    prd_mask = probs > 0.5
    prd_frac = prd_mask.mean()
    prd_mean_prob = probs.mean()
    prd_max_prob = probs.max()
    prd_longest = max((len(list(g)) for k, g in groupby(prd_mask) if k), default=0)

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
        hydro_mean = region_df['HYDRO'].mean()
    else:
        charge_mean = np.nan
        hydro_mean = np.nan

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

def build_full_feature_matrix():
    missing_files = []
    feature_sources = {}

    print("Building base feature matrix...")
    base_df = build_feature_matrix(DATA_DIR)
    for col in base_df.columns:
        feature_sources[col] = 'standard_parsers'

    if os.path.exists(LLPSDB_CSV):
        print("Loading LLPSDB labels...")
        llpsdb_df = pd.read_csv(LLPSDB_CSV)
        llpsdb_df.set_index('protein_id', inplace=True)
        for col in llpsdb_df.columns:
            feature_sources[col] = 'llpsdb'
    else:
        print(f"WARNING: LLPSDB file '{LLPSDB_CSV}' not found. Will create dummy labels.")
        llpsdb_df = None

    print("Predicting secondary structure and solvent accessibility...")
    protein_sequences = {}
    for pid in base_df.index:
        fasta_path = os.path.join(FASTA_DIR, f'{pid}.fasta')
        if os.path.exists(fasta_path):
            with open(fasta_path) as f:
                lines = f.readlines()
                seq = ''.join([l.strip() for l in lines if not l.startswith('>')])
                protein_sequences[pid] = seq
        else:
            missing_files.append(f"FASTA missing for protein {pid}: {fasta_path}")

    try:
        netsurfp_df = predict_netsurfp_features(protein_sequences)
        for col in netsurfp_df.columns:
            feature_sources[col] = 'netsurfp'
    except Exception as e:
        print(f"WARNING: NetSurfP prediction failed: {e}")
        netsurfp_df = pd.DataFrame(0, index=base_df.index, columns=[f'netsurfp_dummy_feature'])
        feature_sources['netsurfp_dummy_feature'] = 'netsurfp_dummy'

    print("Merging features...")
    merged_df = base_df.join(netsurfp_df, how='left')
    if llpsdb_df is not None:
        merged_df = merged_df.join(llpsdb_df, how='left')

    if 'llps_label' not in merged_df.columns:
        print("Assigning dummy labels...")
        merged_df['llps_label'] = 0

    feature_cols = [col for col in merged_df.columns if col != 'llps_label']
    merged_df[feature_cols] = merged_df[feature_cols].fillna(0)

    print(f"Saving full feature matrix to {OUTPUT_CSV}...")
    merged_df.to_csv(OUTPUT_CSV)
    merged_df.drop(columns=['llps_label']).to_csv(OUTPUT_CSV.replace('.csv', '_X.csv'))
    merged_df['llps_label'].to_csv(OUTPUT_CSV.replace('.csv', '_y.csv'))

    if missing_files:
        print(f"Saving missing file report to {MISSING_LOG}...")
        with open(MISSING_LOG, 'w') as f:
            f.write('\n'.join(missing_files))

    print(f"Saving feature origin log to {ORIGIN_LOG}...")
    with open(ORIGIN_LOG, 'w') as f:
        for feature, origin in feature_sources.items():
            f.write(f"{feature}\t{origin}\n")

    print("Matrix generation complete.")
    return merged_df

def split_and_save_train_test(df, output_dir, label_column='llps_label', test_size=0.2, random_state=42):
    os.makedirs(output_dir, exist_ok=True)
    X = df.drop(columns=[label_column])
    y = df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'))
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'))
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'))
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'))

    print(f"Train/test split saved to {output_dir}")
    return X_train, X_test, y_train, y_test

```






































































