Scroll to 'About parse_saps()' section below for info on code and annotated code block at the bottom

1. Paste FASTA
2. Use these parameters for each entry:



(We're counting His as a positive residue to reflect cytoplasmic conditions.)

3. Title job: `proteinID`
4. Submit > View Results 
5. Click the tab for 'Result Files'
6. Download the file 'Tool Output' and rename as: `proteinID_saps`


# Sanity Check: parse_saps.py
`parse_saps() function`: heterogenous named values → spot‑check a few known keys

Run Using:
```python
python parse_saps.py
```

parse_saps.py
```python
if __name__ == "__main__":
    # Update this line w the path to sample SAPS file
    sample_file = "tests/sample_protein_saps.txt"

    expected = {
        'saps_aa_A': 0.052,
        'saps_aa_C': 0.057,
      
        'saps_pct_positively_charged': 0.171,
        'saps_pct_negatively_charged': 0.090,
    }

    # Run parse_saps() function
    features = parse_saps(sample_file)
    print("Parsed features:")
    for k, v in sorted(features.items()):
        print(f"  {k}: {v}")

    # Check these (output) against expected values below
    print("\nSanity check:")
    for key, exp in expected.items():
        actual = features.get(key)
        if actual is None:
            print(f"  Missing feature: {key}")
        elif abs(actual - exp) < 1e-3:
            print(f"  {key} = {actual:.3f} (expected {exp:.3f})")
        else:
            print(f"  {key} = {actual:.3f} (expected {exp:.3f})")

```

If using p_cbx2_iso2_saps.txt as sample, it should show:
```python
# Expected values (from manual inspection of the sample):
    #   - For amino acid A: 5.2% → 0.052
    #   - For C: 5.7% → 0.057
    #   - Positive charge (KRH): 17.1% → 0.171
    #   - Negative charge (ED): 9.0% → 0.090
    #   - (If hydropathy present) mean hydropathy: e.g., 0.45
```

### ==About parse_saps() function==
How parser maps SAPS data to feature dataset:
1. **Amino acid composition**:
    - Uses `comp_re` to catch lines like `A : 11( 5.2%);` and stores the percentage as a fraction.
2. **Charge percentages**:
    - Catches “KRH” (positively charged) and “ED” (negatively charged) with `charge_re`.
3. **Hydropathy**:
    - Looks for a line containing `Hydrophobicity (mean): X` via `hydro_re`.


### ==About parse_saps.py Sanity Check==
To verify that `parse_saps()` correctly extracts features from a SAPS output, the sanity check script:
1. Loads a known sample file: `p_cbx2_iso2_saps.txt`
2. Runs the** **`parse_saps()` function from `parser_v[#].py` code
3. Prints out the resulting feature dictionary
4. Then, we manually check the printed values against the actual values from the file (I've already organized what the correct answers are below the script)



## Pre-Processing Considerations 
SAPS output will need to be formatted to ==.csv== input for ML:

1. Parse SAPS output from plain-text output to a fixed-length vector for each protein
    - Python script to extract relevant values: [saps_parser.py] 


2. Normalize ==(optional: decision needed)==
    - Modify length-dependent values to scalable data (i.e. # of amino acids to frequency/ratio/percentage) 

- [saps_parser_normalizer.py] parses SAPS & includes *only *Gly, Ser, Pro, Tyr, Arg % content .csv output file

- [saps_parser_normalizer20.py] parses SAPS &  includes *all 20 *amino acids % content in .csv output file


3. Format as a feature matrix: ==.csv==

- Each Row = One Protein
- Each Column = SAPS Output as 'Feature'

| Feature | Relevance |
|---|---|

| | |
|---|---|
| %Gly, %Ser, %Arg, %Tyr | Often higher in LLPS-prone prots |

| | |
|---|---|
| Net charge / residue | Affects electrostatic interactions |

| | |
|---|---|
| Sequence entropy | More entropy → more disordered |

| | |
|---|---|
| Max length of homorepeats | Repeats can drive phase separation |

| | |
|---|---|
| Hydrophobicity | Important for weak interactions |

| | |
|---|---|
| Polar:Non-polar ratio | Interaction propensity |

| | |
|---|---|
| Acidic/basic residue ratio | May influence condensation  |

| | |
|---|---|
| Length | Longer may be more likely to LLPS |












































