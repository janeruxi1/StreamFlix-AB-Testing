# Streamlit demo

Interactive companion to the repository: see the real results, change the assumptions behind the
decision, and plan or analyse a test of your own without reading code.

## Run locally

From the project root:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## What it does

1. **📧 The e-mail test** (Hillstrom): the decision, each e-mail's effect on spend, purchases and
   visits with a regression-adjustment toggle, a Bayesian view of the purchase and visit rates,
   and margin and cost sliders that show when the decision would change. Read from the committed
   records in `reports/metrics/`, because the raw file is not redistributed.
2. **🏆 Is this winner real?** (Upworthy): any of 17,703 headline tests, with each headline's
   observed click-through against its empirical-Bayes shrunk estimate. Read from the committed
   `data/upworthy/confirmatory.parquet`.
3. **📐 Plan a test:** baseline rate, smallest lift worth detecting, alpha, power and daily traffic
   give users per arm, runtime and a sample-size curve.
4. **🔬 Analyse your own results:** users and conversions per arm give a frequentist and a
   Bayesian readout, the posterior and the expected cost of each decision.
5. **🎬 StreamFlix case study:** the simulated experiment from this project's first version,
   recomputed from `data/experiment.csv`.

## Live demo

[janeruxi1-ab-testing-project.streamlit.app](https://janeruxi1-ab-testing-project.streamlit.app/)

## How it stays correct

The app imports from `src/analysis/`, the same code the notebooks use and the tests check, and
`tests/test_app_smoke.py` runs the app and checks it shows the reported results.
