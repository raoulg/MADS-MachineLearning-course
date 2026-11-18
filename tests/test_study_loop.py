"""The loop figure a notebook draws comes from the page students read, whichever of its
addresses the notebook names."""

import importlib.util
from pathlib import Path

import pytest

LESSON = Path(__file__).resolve().parents[1] / "notebooks/1_representation_and_head"


@pytest.fixture(scope="module")
def plotting():
    spec = importlib.util.spec_from_file_location("plotting", LESSON / "plotting.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "page", ["your-study.html", "lesson1-your-study.html", "lesson2-your-study.html"]
)
def test_every_address_of_the_page_draws_the_nine_steps(plotting, page: str) -> None:
    figure = plotting.study_loop(page).data
    for step in range(1, 10):
        assert f'href="{plotting.PAGES_URL}your-study.html#step-{step}"' in figure


def test_a_page_without_a_loop_is_refused(plotting) -> None:
    with pytest.raises(ValueError, match="holds no loop figure"):
        plotting.study_loop("notatie.html")
