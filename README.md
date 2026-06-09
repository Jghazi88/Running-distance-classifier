# Running Distance Classifier

A scikit-learn multi-class classifier that predicts a person's maximum race
distance — **5K**, **10K**, **Half Marathon**, or **Marathon** — from five
health and fitness features.

The target is an **ordinal** outcome (5K < 10K < Half Marathon < Marathon),
which is reflected in the confusion matrix: nearly every misclassification
lands on a *neighbouring* class rather than a distant one.

## Features

The model is trained on five numeric features per person:

| Feature         | Description                                  |
| --------------- | -------------------------------------------- |
| `vo2_max`       | Aerobic capacity (mL/kg/min)                 |
| `weekly_miles`  | Average running mileage per week             |
| `mile_time_min` | Best mile time (minutes)                     |
| `years_running` | Years of consistent running experience       |
| `body_fat_pct`  | Body fat percentage                          |

The target column `distance_class` is integer-encoded `0..3`, decoded back
to a human-readable label at inference time via the `class_order` list.

## Results

| Model               | Test accuracy |
| ------------------- | ------------- |
| Logistic Regression | 0.785         |
| Random Forest       | 0.7475        |
| Gradient Boosting   | 0.7175        |

Feature importances from the Gradient Boosting model rank `vo2_max` and
`weekly_miles` as the dominant predictors, followed by `body_fat_pct`,
`years_running`, and `mile_tim_min`.

## Quickstart

```bash
# 1. Clone
git clone https://github.com/<your-username>/running-distance-classifier.git
cd running-distance-classifier

# 2. Install dependencies (a virtual environment is recommended)
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook running_distance_classifier.ipynb
```

Run all cells (`Cell → Run All`) to regenerate the synthetic dataset, train
the three models, evaluate them, and try `predict_distance()` on a sample
person.

## Project structure

```
.
├── running_distance_classifier.ipynb   # main notebook (data, models, demo)
├── requirements.txt                    # Python dependencies
├── README.md                           # this file
└── .gitignore                          # ignored files
```

## How it works

1. **Synthetic data.** 2,500 rows are generated with a fixed random seed
   (`RANDOM_STATE = 42`) so results are fully reproducible. A latent
   "endurance fitness" score is built from the features, then split at
   its quartiles into the four ordered classes.
2. **Train / test split.** Stratified 80 / 20 split, holding out 500 rows
   for evaluation.
3. **Three models.** Logistic Regression (standardized), Random Forest,
   and Gradient Boosting are trained and compared on the test set.
4. **Single-person inference.** `predict_distance(model, person_dict)`
   packs one person into a 2-D array of shape `(1, 5)`, asks the model
   for an integer class index, and decodes it back to a label.

## Notes

- The dataset is **fully synthetic**. No real athlete data is used, and
  the model should not be relied upon for any health, training, or
  medical decision.
- Because the data is procedurally generated, accuracy and feature
  importances are deterministic for a given seed.

## License

Released under the MIT License. See `LICENSE` if one is added to the repo.
