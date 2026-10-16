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

Two modes:

1. **Pre-experiment design** — slider-driven sample-size calculator. Set baseline rate, MDE, alpha, power, daily traffic. Outputs required users per arm, total sample size, runtime in days, and a sample-size-vs-MDE curve highlighting your selected point.

2. **Post-experiment analysis** — paste in conversion counts for control and treatment arms. Outputs:
   - Absolute and relative lift
   - 95% confidence interval
   - Frequentist p-value
   - Bayesian P(treatment > control), 95% credible interval, expected loss of shipping
   - Posterior distribution chart

## Live demo

Deployed on Streamlit Community Cloud: [janeruxi1-ab-testing-project.streamlit.app](https://janeruxi1-ab-testing-project.streamlit.app/)

## How it stays correct

The app imports directly from `src/analysis/` — the same code path the analysis notebooks and the 62 pytest unit tests use. The demo therefore runs the same tested functions as the analysis.
