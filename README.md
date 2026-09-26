# Freight Rate Prediction

Predicts `posted_rate` (per load) from route, equipment, weight, date, and
market-signal features, and applies the trained model to
`data/validation.csv` and the fixed December lane in
`data/december_chart_inputs.csv`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To rebuild the Word report (`report/Assessment_Report.docx`), you also need Node:

```bash
cd report
npm install
node build_report.js   # reads outputs/scorer_results/candidate_december.png
cd ..
```

## Run

```bash
python3 src/train.py             # cleans data, compares models on a time-based
                                  # holdout, refits the chosen model on all
                                  # labeled data, saves outputs/model.joblib
python3 src/predict.py           # scores data/validation.csv ->
                                  # outputs/validation_predictions.csv
python3 src/december_chart.py    # forecasts the fixed December lane ->
                                  # outputs/december_chart_inputs_filled.csv
python3 score.py --predictions outputs/validation_predictions.csv \
    --december-predictions outputs/december_chart_inputs_filled.csv \
    --output-dir outputs/scorer_results
                                  # validates both files and renders the
                                  # official chart -> outputs/scorer_results/candidate_december.png
```

`train.py` must be run before `predict.py` / `december_chart.py`, which
both load `outputs/model.joblib`. `score.py` (provided with the
assessment) only validates format and plots whatever `predicted_rate`
values it's given -- it does not fit or run any model itself, so
`december_chart.py` is what actually produces the December predictions.

## Repo layout

```
data/                       input CSVs (as provided)
src/
  features.py                cleaning + feature engineering (shared)
  train.py                   time-based split, model comparison, final fit
  predict.py                 scores validation.csv
  december_chart.py          scores/plots the fixed December lane
score.py                      provided scorer: validates output files, renders the official December chart
outputs/
  model.joblib                trained model + imputation stats + column order
  metrics.json                holdout metrics for all models tried
  validation_predictions.csv  final deliverable: load_id, predicted_rate
  december_chart_inputs_filled.csv   completed December template (fed to score.py)
  scorer_results/candidate_december.png   official December chart (score.py output)
report/                       write-up (this assessment's report deliverable)
```

## Notes / assumptions

- `december_chart_inputs.csv` has no `market_index` / `quote_signal` columns
  (only the date varies across its 31 rows), but the trained model needs
  both. `december_chart.py` forecasts them with a small day-of-week +
  day-of-year harmonic regression fit on the historical daily averages,
  then scores the fixed lane for each December date.
- Full approach, validation strategy, and findings are in `report/`.
