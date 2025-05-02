Code handles our txt files
### INPUT
_saps.txt files
_pondr.txt files
_plaac.txt files
_string.txt files

### OUTPUT
`llps_feature_matrix.csv` 
Pandas DataFrames feature matrix with rows = proteins ; columns = features

```python
import os
import re
import pandas as pd
import numpy as np
from glob import glob

# ==== PONDR PARSER ====
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

# ==== SAPS PARSER ====
def parse_saps(filepath):
    features = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if "Charge" in line and "overall" in line:
            match = re.search(r'overall\s+([-+]?\d*\.\d+|\d+)', line)
            if match:
                features['saps_net_charge'] = float(match.group(1))
        if "Hydrophobicity" in line:
            match = re.search(r'Hydrophobicity.*?mean:\s*([-+]?\d*\.\d+|\d+)', line)
            if match:
                features['saps_mean_hydropathy'] = float(match.group(1))
        if "Amino acid composition" in line:
            aa_section = lines[lines.index(line)+2:lines.index(line)+22]
            for aa_line in aa_section:
                parts = aa_line.strip().split()
                if len(parts) >= 2:
                    aa, freq = parts[0], float(parts[1])
                    features[f'saps_aa_{aa}'] = freq
    return features

# ==== PLAAC PARSER ====
def parse_plaac(filepath):
    features = {}
    with open(filepath, 'r') as f:
        text = f.read()
    prion_score_match = re.search(r'PLAAC Score:\s*([\d\.]+)', text)
    if prion_score_match:
        features['plaac_score'] = float(prion_score_match.group(1))
    domain_match = re.search(r'Domain Range:\s*(\d+)-(\d+)', text)
    if domain_match:
        features['plaac_domain_length'] = int(domain_match.group(2)) - int(domain_match.group(1)) + 1
    return features

# ==== STRING PARSER ====
def parse_string(filepath):
    df = pd.read_csv(filepath, sep='\t')
    features = {
        'string_ppi_count': len(df),
        'string_mean_score': df['combined_score'].mean(),
        'string_high_conf_interactors': len(df[df['combined_score'] > 700])
    }
    return features

# ==== MASTER FUNCTION ====
def build_feature_matrix(data_dir):
    proteins = [os.path.basename(f).split('_')[0] for f in glob(os.path.join(data_dir, 'pondr', '*.txt'))]
    feature_rows = []

    for pid in proteins:
        row = {'protein_id': pid}
        try:
            row.update(parse_pondr(os.path.join(data_dir, 'pondr', f'{pid}_pondr.txt')))
        except Exception as e:
            print(f"[PONDR] Error in {pid}: {e}")

        try:
            row.update(parse_saps(os.path.join(data_dir, 'saps', f'{pid}_saps.txt')))
        except Exception as e:
            print(f"[SAPS] Error in {pid}: {e}")

        try:
            row.update(parse_plaac(os.path.join(data_dir, 'plaac', f'{pid}_plaac.txt')))
        except Exception as e:
            print(f"[PLAAC] Error in {pid}: {e}")

        try:
            row.update(parse_string(os.path.join(data_dir, 'string', f'{pid}_string.txt')))
        except Exception as e:
            print(f"[STRING] Error in {pid}: {e}")

        feature_rows.append(row)

    df = pd.DataFrame(feature_rows)
    df.set_index('protein_id', inplace=True)
    df = df.fillna(0)
    return df

# ==== EXAMPLE USAGE ====
if __name__ == "__main__":
    data_dir = 'data'
    feature_matrix = build_feature_matrix(data_dir)
    feature_matrix.to_csv("llps_feature_matrix.csv")
    print("Saved feature matrix with shape:", feature_matrix.shape)

```
































