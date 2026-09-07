# Tabular Regression Challenge

ML pipeline for tabular regression over 53 anonymized features.

## Findings & Methodology
- Feature screening and scatter analysis revealed that 51 features are synthetic noise.
- The target follows the analytical formulation: `target = (feature_6)^2 + feature_7`.
- Standard gradient boosted trees achieve ~0.02 RMSE due to discrete splits on curves. A regularized linear pipeline with an explicit quadratic term yields near-zero error (~4.8e-10 RMSE).

## Execution
```bash
# Train model (5-Fold CV & artifact export)
python3 train.py

# Generate predictions for hidden_test.csv
python3 predict.py
```
