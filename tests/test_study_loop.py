"""The loop figure of the page students read draws the nine steps in both layouts, and a
notebook draws it from that page, whichever of its addresses the notebook names. Each
lesson's block on that page opens with "Your research" and three prompts of its own lesson."""

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


PROMPTS = ["Start my study", "Build my page and interview me", "Submit my runs"]
RAW = (
    "https://raw.githubusercontent.com/raoulg/MADS-MachineLearning-course/master/docs/"
)


def lesson_block(html: str, lesson: int) -> str:
    block = re.search(
        rf'<section class="lesson" id="lesson-{lesson}">.*?</section>', html, re.DOTALL
    )
    assert block is not None
    return block.group(0)


@pytest.mark.parametrize("lesson", [1, 2])
def test_each_lesson_block_opens_with_your_research_and_its_three_prompts(
    lesson: int,
) -> None:
    block = lesson_block(PAGE.read_text(), lesson)
    after_title = block[block.index("</h3>") :]
    assert re.match(
        r'</h3>\s*<div class="research">\s*<h4>Your research</h4>', after_title
    )
    titles = re.findall(r'<div class="prompt-head"><span>([^<]*)</span>', block)
    assert titles == PROMPTS
    prompts = re.findall(r"<pre>([^<]*)</pre>", block)
    assert len(prompts) == 3
    other = 3 - lesson
    for prompt in prompts:
        assert f"lesson {lesson}" in prompt and f"lesson {other}" not in prompt
    start, explain, submit = prompts
    assert "docs/your-study.md" in start and f"{RAW}your-study.md" in start
    assert f"make page LESSON={lesson}" in explain and f"{RAW}your-study.md" in explain
    assert "docs/submit-runs.md" in submit and f"{RAW}submit-runs.md" in submit


def test_the_page_says_how_it_counts_once_and_never_says_homework() -> None:
    html = PAGE.read_text()
    assert html.count('id="how-it-counts"') == 1
    assert html.count("above 5.5") == 1
    assert "homework" not in html.lower()
    assert f'href="{RAW}' not in html  # the raw URLs are prompt text, not links
