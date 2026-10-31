"""Every headline number in the README and the decision memo is read from the recorded results.

Each check formats a value from reports/metrics/ the way the docs write it and asserts that the
text appears in the doc. If a notebook's result changes, these fail until the docs are updated.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = {name: (ROOT / path).read_text(encoding="utf-8")
        for name, path in {"readme": "README.md", "memo": "reports/decision_memo.md"}.items()}


def rec(name):
    return json.loads((ROOT / "reports" / "metrics" / f"{name}.json").read_text(encoding="utf-8"))


C01, C02, C03 = rec("01_email_test_checks"), rec("02_email_test_readout"), rec("03_email_test_targeting")
C04, C05 = rec("04_sample_ratio_confirmatory"), rec("05_winners_confirmatory")


def money(x):
    return f"{'+' if x >= 0 else '−'}${abs(x):.2f}"


def signed(x):
    return f"{'+' if x >= 0 else '−'}{abs(x):.2f}"


def eff(metric, i):
    return C02["effects"][metric][i]


def dollars(x):
    return f"{'−' if x < 0 else ''}${abs(x):,.0f}"


MENS, WOMENS, HEAD = (eff("spend_capped", i) for i in range(3))
DESIGN = {d["metric"]: d for d in C01["design"]}
HOLD5 = next(h for h in C02["rollout"]["holdback"] if h["holdback_share"] == 0.05)

CLAIMS = [
    # Part 1
    ("both", f"{money(MENS['diff'])}"),
    ("memo", f"**{money(MENS['diff'])}** [{signed(MENS['ci_low'])}, {signed(MENS['ci_high'])}]"),
    ("memo", f"{money(WOMENS['diff'])} [{signed(WOMENS['ci_low'])}, {signed(WOMENS['ci_high'])}]"),
    ("memo", f"{money(HEAD['diff'])} [{signed(HEAD['ci_low'])}, {signed(HEAD['ci_high'])}]"),
    ("readme", f"{money(MENS['diff'])} spend per customer [{signed(MENS['ci_low'])}, {signed(MENS['ci_high'])}]"),
    ("readme", f"by ${HEAD['diff']:.2f} [{signed(HEAD['ci_low'])}, {signed(HEAD['ci_high'])}]"),
    ("readme", f"{dollars(C02['value']['mens']['value_per_1000'])} per 1,000"),
    ("memo", f"**{dollars(C02['value']['mens']['value_per_1000'])}** ["),
    ("memo", f"{dollars(C02['value']['mens']['value_per_1000_ci'][0])}, {dollars(C02['value']['mens']['value_per_1000_ci'][1])}]"),
    ("memo", f"{dollars(C02['value']['womens']['value_per_1000'])} [{dollars(C02['value']['womens']['value_per_1000_ci'][0])}, "
             f"{dollars(C02['value']['womens']['value_per_1000_ci'][1])}]"),
    ("memo", f"| {C02['value']['mens']['breakeven_margin']:.0%} |"),
    ("memo", f"| {C02['value']['womens']['breakeven_margin']:.0%} |"),
    ("memo", f"visits (+{eff('visit', 0)['diff'] * 100:.1f} points)"),
    ("memo", f"purchases (+{eff('conversion', 0)['diff'] * 100:.2f} points)"),
    ("memo", f"p = {C01['srm']['p_value']:.2f}"),
    ("readme", f"p = {C01['srm']['p_value']:.2f}"),
    ("memo", "{:,} / {:,} / {:,}".format(*C01["srm"]["counts"].values())),
    ("both", f"{money(C03['confirmation']['gain'])} [{money(C03['confirmation']['ci'][0])}, "
             f"{money(C03['confirmation']['ci'][1])}]"),
    ("memo", f"{round(C03['shrinkage'][C03['chosen']]['discovery gain over single'] * 100)} cents"),
    ("both", f"${DESIGN['spend']['mde_at_this_size']:.2f}"),
    ("both", f"${DESIGN['spend']['effect_worth_detecting']:.2f}"),
    ("memo", f"{DESIGN['spend']['n_per_arm_needed'] / 1000:.1f}".replace(".", ",") + "00 per arm"),
    ("memo", f"{HOLD5['customers_held_back']:,} customers"),
    ("memo", f"${round(HOLD5['margin_given_up_per_send'], -1):,.0f} of margin"),
    ("memo", f"over {HOLD5['sends_to_detect_half_the_lift']} sends"),
    ("memo", f"{C02['rollout']['next_test']['n_per_arm'] // 1000:,},000 customers per arm"),
    ("memo", f"about {C02['rollout']['next_test']['sends_needed']:.1f} sends"),
    # Part 2
    ("both", f"{C04['tests']:,}"),
    ("both", f"**{C04['flag_rate_inside']:.0%}**"),
    ("both", f"**{C04['flag_rate_outside']:.1%}**"),
    ("memo", f"({C04['ratio']:.0f} times as often)"),
    ("memo", f"**{C05['tau']:.0%}**"),
    ("memo", f"95% CI {C05['tau_ci_tests'][0]:.0%}–{C05['tau_ci_tests'][1]:.0%}"),
    ("memo", f"lift of {C05['winners']['lift_a_mean']:.2f}"),
    ("memo", f"it shows {C05['winners']['lift_b_mean']:.2f}"),
    ("both", f"{C05['winners']['survival']:.0%}"),
    ("memo", f"{C05['winners']['same_winner_rate']:.0%} of tests, against {C05['winners']['chance_same_winner']:.0%}"),
    ("memo", f"{C05['winners']['shrunk_lift_mean']:.2f} against the {C05['winners']['lift_b_mean']:.2f} measured"),
    ("both", f"+{C05['gain_vs_random_relative']:.0%}"),
]


@pytest.mark.parametrize("where,text", CLAIMS, ids=[c[1][:40] for c in CLAIMS])
def test_doc_quotes_the_recorded_result(where, text):
    for name in (["readme", "memo"] if where == "both" else [where]):
        assert text in DOCS[name], f"{name} should contain {text!r}"


def test_records_are_from_real_data_and_the_holdout_agrees():
    for r in (C01, C02, C03, C04, C05):
        assert r["source"] == "real"
    assert C04["sample"] == "confirmatory" and C04["confirmed"]
    assert C05["holdout_agrees"]


def test_decisions_match_the_docs():
    assert C02["decision"] == "mens" and "men's merchandise e-mail to every" in DOCS["memo"]
    assert C03["recommend_targeting"] is False and "do not target" in DOCS["memo"]
