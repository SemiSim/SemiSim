# SemiSim
A protoype AI Monte-Carlo sim block used to predict LLPS interactions
intented for CHEM-4640/5640 course at The University of Colorado Denver. The SEMISIM model was created to progress in vivo biomolecular LLPS research through the use if AI – Monte Carlo simulations.​
By using AI simulations, we hope to speed up​ and possibly revolutionize research to promote collaboration and help scientists investigate LLPS more in depth.

## What can SemiSim do?
The **"SemiSim"** is a deep learning pipeline designed to predict liquid-liquid phase separation (LLPS) in proteins by analyzing both sequence and structural features. It processes protein sequences by converting amino acid letters into indexed embeddings and then feeds these through an LSTM layer to capture sequential dependencies. In parallel, it incorporates five structural features derived from NetSurfP (e.g., disorder probability and secondary structure tendencies), which are normalized and used as additional inputs. These two feature streams are concatenated and passed through fully connected layers to output a probability of LLPS using a sigmoid-activated classification layer.

To address class imbalance, the notebook includes a custom focal loss function and also supports the use of class weighting during model training. The dataset is cleaned, balanced, and prepared by converting labels to binary (yes/no → 1/0), encoding sequences, and standardizing structural features. Model evaluation is performed using 5-fold stratified cross-validation, reporting key metrics such as accuracy, precision, recall, and ROC AUC scores after each fold. Overall, SemiSim serves as a robust framework for LLPS prediction, integrating biological feature engineering with deep learning techniques for classification.
