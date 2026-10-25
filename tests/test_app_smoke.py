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
