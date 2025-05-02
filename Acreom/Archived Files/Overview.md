
1. DATA HARVESTING
    1. Primary Structure & Sequence Data
    2. Physiochemical Properties
    3. Structure & Interaction
    4. Chains & Domains 

#                              ⬇︎
Harvested data has to be converted into machine-readable formats 
#                              ⬇︎
2. PREPPING DATA FOR ML
    1. Primary Structure & Sequence Data ➺ convert with One-Hot Encoding or Embedding (ProBERT, ESM models, etc.)
    2. Physiochemical Properties ➺ convert to Numerical Vectors  ➺  Format vectors as a Feature Matrix
        1. ==SAPS== was converted to numerical feature vector of fixed length, and then formatted as a feature matrix ==.csv== file using `saps_parser.py` code. 
    3. If using GNN for PPI ➺ convert to Graphed Representations
    4. Other (Remaining) Data ➺ convert to Custom Feature Vectors ~~for traditional ML models (XGBoost, SVM, etc.)~~

#                              ⬇︎

Select model architecture & algorithm framework for our objective & data input format(s):
#                              ⬇︎
3. MODEL ARCHITECTURE = LTSM maybe?
4. FRAMEWORK = KERAS~~ or PYTORCH ~~

#                              ⬇︎



































































