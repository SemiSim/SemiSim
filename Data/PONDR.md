Scroll to 'About parse_pondr()' section below for info on code and annotated code block at the bottom

To get PONDR output: .txt
1. Open PONDR
2. ☑︎ VLXT
3. Type the "Protein Name" as its assigned protein ID `proteinID`
4. In a new tab, open NCBI RefSeq > Proteins > Search > Click 'FASTA'

NOTE: Exclude 'partial' sequence entries! Will be indicated here on the FASTA display page.
3. Copy & Paste into PONDR > ☑︎ all output boxes EXCEPT 'Graphic'
4. Submit Query

5. Copy & Paste output into ==.txt==
6. Save as `proteinID_pondr.txt` in folder `./pondr`


# Sanity Check: parse_pondr() 
`parse_pondr()`: homogeneous list → uniform checks against computed stats
```python
scores = []
with open('protein_pondr.txt') as f:
    for line in f:
        if re.match(r'^\d+\s+\w\s+[\d\.]+', line):
            parts = line.split()
            scores.append(float(parts[2]))

import numpy as np
from itertools import groupby

print("mean:", np.mean(scores))
print("max:", np.max(scores))
mask = [s>0.5 for s in scores]
print("frac:", sum(mask)/len(mask))
print("longest:", max((len(list(g)) for k,g in groupby(mask) if k), default=0))

```

### ==About parse_pondr() function==
Here’s how each output field maps from the _pondr.txt files:
1. `scores` list
    - The function’s regex `r'^\d+\s+\w\s+[\d\.]+'` was made to match lines in the **“**PREDICTOR VALUES**”** section
    - It splits each matching line and takes `parts[2]`, which is the per‑residue PONDR VLXT score
2. `pond_mean_disorder`
    - Average of all PONDR VLXT scores in the bottom table
3. `pond_max_disorder`
    - Max PONDR VLXT score from the table
4. `pond_disordered_frac`
    - Fraction of rediues with predicted disorder scores > 0.5.
    - Ex: our file `p_cbx2_iso2_pondr.txt` has 90 predicted disordered residues (PONDR score > 0.5), so `90/211 ≈ 0.4265` 
5. `pond_longest_disordered`
    - Length of the longest contiguous run of scores > 0.5. In your header it says the longest region is 45 residues, so this should also come out as 45.













































