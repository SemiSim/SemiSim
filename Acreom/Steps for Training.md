Step 1. Use K-Fold Validation
- At bottom of code, make sure the following is present:

```python
run_kfold(sequence_data, X_scaled, y, n_splits=5)
```
-Then, comment out the other one `(train_model)`.
- This will:
    - Train 5 models (folds)
    - Print fold-by-fold classification report, confusion matric, and ROC AUC score
    - **NOT save a model**** yet**


Step 2. To Train and Save a Model
- If happy with performance, comment out `run_kfold` and uncomment this:

```
model = train_model(sequence_data, X_scaled, y, epochs=20, batch_size=32)
```
- This will:
    - Split into 70% train/15% val/ 15% test
    - Train model
    - Save the trained model as `semisim.h5`
    - Print a final test set evaluation
