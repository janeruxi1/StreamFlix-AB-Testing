"""Smoke tests: the Streamlit app runs end to end and shows the documented result."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_headline_matches_report(app):
    values = [m.value for m in app.metric]
    assert "+2.54pp" in values                      # README / memo headline
    assert any("+2.02 to +3.06pp" in v for v in values)


def test_analyse_tab_prefilled_with_experiment_counts(app):
    assert app.number_input(key="nc").value == 50_461
    assert app.number_input(key="nt").value == 49_539


def test_widgets_update_without_errors(app):
    app.slider(key="mde").set_value(0.5).run()
    app.radio(key="seg_by").set_value("Country").run()
    app.number_input(key="kt").set_value(10_726).run()   # no lift
    assert not app.exception, app.exception


def test_email_tab_shows_the_reported_decision(app):
    values = [m.value for m in app.metric]
    assert "+$0.77" in values and "$207" in values
    assert any("men's e-mail to everyone" in el.value for el in app.success)


def test_email_controls_and_winner_explorer_run(app):
    app.toggle(key="email_adjust").set_value(False).run()
    app.radio(key="email_metric").set_value("Purchase rate").run()
    app.slider(key="email_margin").set_value(0.10).run()
    assert any("neither e-mail pays" in el.value or "includes a loss" in el.value
               for el in list(app.error) + list(app.warning))
    app.button(key="winner_next").click().run()
    assert not app.exception, app.exception
