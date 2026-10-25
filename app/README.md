# Streamlit Demo

Interactive companion to the StreamFlix A/B testing project. Lets visitors plug in their own numbers and get the full pre-experiment design or post-experiment analysis without reading code.

## Run locally

From the project root (one level up):

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## What it does

Three tabs:

1. **🏁 The StreamFlix result** — the ship decision, headline lift with its confidence interval and Bayesian probability, the checks a reviewer asks about (sample ratio, the lift without Android, day-7 activity and page-load guardrails), and the lift by device, country, source and new vs returning users. Recomputed from `data/experiment.csv` when the app starts.
2. **📐 Plan a test** — set the baseline rate, smallest lift worth detecting, alpha, power and daily traffic; get users per arm, runtime and a sample-size curve, plus the smallest lift a fixed audience can detect.
3. **🔬 Analyse your own results** — enter users and conversions per arm (pre-filled with this experiment) for the lift, confidence interval, Bayesian probability of a win, posterior chart and the expected cost of each decision.

## Live demo

Deployed on Streamlit Community Cloud: [janeruxi1-ab-testing-project.streamlit.app](https://janeruxi1-ab-testing-project.streamlit.app/)

## How it stays correct

The app imports directly from `src/analysis/` — the same code path the analysis notebooks and the pytest suite checks; `tests/test_app_smoke.py` also runs the app itself and checks it shows the reported headline. The demo therefore runs the same tested functions as the analysis.
