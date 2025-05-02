
- [x] Decide on incorporation of PhaseSepDB.csv 


- [x] Find substitute for secondary structure & solvent accessibility (formerly sourced from PDB, DSSP) = NetSurfP 
- [x] Script for batching FASTA sequences to NetSurfP 2.0 web API  
- [x] Modular integration code for NetSurfP outputs into full script 


- [x] Do we want a parse FASTA function? 

No: Input for users will likely be a FASTA file so, wanna train model on FASTA. Fasta handling comes in during AI training script 

- [x] See if we can find a way to parse the BioGRID data dumps too large to open -not worth it  
- [x] Find an alternative source for PTM data if we can't get BioGRID ptm file to open (the ptm_rel file opens fine) - mod later, if time 


- [x] Write code block importing labels from LLPSDB~~ / PhaseSepDB ~~to map feature matrix data with 


- [x] ~~update PDB parse function to handle .cif files instead (del?*)~~ 

- [x] Assemble sample directory & files 
- [x] Set up yaml + config file ; Mod script to code for yaml integ instead of hardcoding filepaths  
- [x] Write script for generating requirements file for GitHub [generate_requirements.py] 



## Complete File Aquisitions:
- [x] FASTA 
- [x] PONDR 
- [x] SAPS 
- [x] PLAAC 
- [x] STRING 
- [x] BioGRID 
- [x] ~~PDB~~ 
- [x] ~~DSSP~~ 
- [x] LLPSDB 


## Parse Functions Built & Master Function Updated:
- [x] PONDR 
- [x] SAPS 
- [x] PLAAC 
- [x] STRING 
- [x] BioGRID (individual files) 
- [x] ~~BioGRID (full DB) *~~ 
- [x] ~~BioGRID (ptm ; ptm_relationships)~~ * 
- [x] ~~PDB~~ done, but deleted 
- [x] ~~DSSP~~ done, but deleted 
- [x] LLPSDB 


## Sanity Check Written:
- [x] parse_pondr.py 
- [x] parse_saps.py 
- [ ] parse_plaac.py 
- [ ] parse_string.py 
- [ ] parse_biogrid.py (individual files) 
- [x] ~~parse_pdb.py~~ 
- [x] ~~parse_dssp.py~~ 
- [ ] parse_llpsdb.py 









































