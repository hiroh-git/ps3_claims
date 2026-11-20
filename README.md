# Problem Set 3: Claims modelling exercise

```
mamba env create
conda activate ps3

pre-commit install
pip install --no-build-isolation -e .
```
## Exercise 3 results

All models use the same fixed train/test split. We divide claim amount
by exposure to express pure premium per insured year, and use exposure
as a weight when fitting and scoring the models.

| Model | Train deviance | Mean CV deviance | Test deviance | Test Gini |
|---|---:|---:|---:|---:|
| Linear Tweedie | 73.7703 | 74.3388 | 73.6024 | 0.237 |
| Quadratic Tweedie | 73.7050 | 74.3166 | 73.5667 | 0.238 |
| Tuned LightGBM | 69.9171 | 73.5254 | 72.8520 | 0.251 |

Adding squared terms for `BonusMalus` and `VehPower` slightly reduced
both training and test deviance. Validation deviance improved in two
of the three folds, so the gain varied across splits.

The best LightGBM settings were a learning rate of 0.03, 250 estimators,
and 16 leaves. LightGBM had a larger gap between training and CV deviance,
but still achieved the lowest test deviance, about 1% below the linear
benchmark, and the highest test Gini among the fitted models.