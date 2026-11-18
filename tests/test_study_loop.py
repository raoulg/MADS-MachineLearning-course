"""The loop figure of the page students read draws the nine steps in both layouts, and a
notebook draws it from that page, whichever of its addresses the notebook names."""

import importlib.util
import re
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


PAGE = LESSON.parents[1] / "presentations/your-study.html"
STEPS = [
    "Design",
    "Build",
    "Baseline",
    "Budget, space",
    "Search",
    "Read",
    "Explain",
    "Share",
    "Build on",
]


@pytest.mark.parametrize("layout", ["loop-wide", "loop-tall"])
def test_both_layouts_of_the_figure_draw_the_nine_steps_in_order(layout: str) -> None:
    html = PAGE.read_text()
    svg = re.search(rf'<svg class="{layout}".*?</svg>', html, flags=re.DOTALL)
    assert svg is not None
    labels = re.findall(r'<text class="t"[^>]*>([^<]*)</text>', svg.group(0))
    assert labels == [f"{n} {step}" for n, step in enumerate(STEPS, 1)]
    links = re.findall(r'href="#(step-\d+)"', svg.group(0))
    assert links == [f"step-{n}" for n in range(1, 10)]


def test_every_step_has_one_card_and_the_figure_says_nine() -> None:
    html = PAGE.read_text()
    for n in range(1, 10):
        assert html.count(f'id="step-{n}"') == 1
    assert 'id="step-10"' not in html
    label = re.search(r'<figure class="loop" aria-label="([^"]*)"', html)
    assert label is not None and label.group(1).startswith("Nine steps")
