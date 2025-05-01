# For the future
**✧ Model Interpretation**: After achieving a reasonable model performance, we need to interpret the model to understand what features (e.g., specific amino acids, secondary structures) contribute most to its LLPS predictions. - In other words, we need to determine what features does our model weigh most heavily in generating its predictions.

We're using SHAP to interpret the trained model & understand which features are weighted most significant contributors to LLPS in algorithm's predictions (unless you have a better idea?)

Install SHAP first:
```bash
pip install shap

```

Then, add this code **after**** **training the model:
```python
import shap

def explain_model(model, dataset):
    model.eval()
    background = dataset.X[:100]  # Use subset for speed
    explainer = shap.Explainer(model, background)
    shap_values = explainer(dataset.X)

    # Plot summary
    shap.summary_plot(shap_values, dataset.X.numpy(), feature_names=[f'feat_{i}' for i in range(dataset.X.shape[1])])

```

To run, use:
```python
explain_model(model, dataset)

```
