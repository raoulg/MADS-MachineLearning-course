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


NOTEBOOK = {
    1: "notebooks/1_representation_and_head/05_exercises.ipynb",
    2: "notebooks/2_kernels_state_and_gates/04_exercise.ipynb",
}


def words(fragment: str) -> str:
    """The text of a fragment: tags dropped, whitespace folded."""
    return " ".join(re.sub(r"<[^>]*>", " ", fragment).split())


@pytest.mark.parametrize("lesson", [1, 2])
def test_each_lesson_block_opens_with_your_research_and_its_three_prompts(
    lesson: int,
) -> None:
    block = lesson_block(PAGE.read_text(), lesson)
    research = block[block.index("</h3>") : block.index("<dl>")]
    assert words(research).startswith("Your research ")
    assert f"To lesson {lesson + 1} you bring your page" in words(research)
    assert f"lesson {lesson + 1} opens on the class's runs" in words(research)
    assert f'/blob/master/{NOTEBOOK[lesson]}"' in research
    assert NOTEBOOK[3 - lesson] not in research
    heads = re.findall(r'<div class="prompt-head">(.*?)</div>', research, re.DOTALL)
    assert [words(head.split("<button")[0]) for head in heads] == PROMPTS
    assert all("<button" in head for head in heads)
    prompts = [
        words(pre) for pre in re.findall(r"<pre>(.*?)</pre>", research, re.DOTALL)
    ]
    assert len(prompts) == 3
    for prompt in prompts:
        assert f"lesson {lesson}" in prompt and f"lesson {3 - lesson}" not in prompt
    start, explain, submit = prompts
    assert "docs/your-study.md" in start and f"{RAW}your-study.md" in start
    assert "ask me first" in start and "in my own words" in start
    assert f"make page LESSON={lesson}" in explain and f"{RAW}your-study.md" in explain
    assert "one question at a time" in explain and "in my own words" in explain
    assert "docs/submit-runs.md" in submit and f"{RAW}submit-runs.md" in submit
    assert "Show me the list of runs first" in submit


def test_the_page_says_how_it_counts_once_and_never_says_homework() -> None:
    html = PAGE.read_text()
    assert html.count('id="how-it-counts"') == 1
    counts = html[html.index('id="how-it-counts"') : html.index('id="per-lesson"')]
    text = words(counts)
    assert "five criteria" in text
    assert "at most 0.5" in text and "5.5" in text
    assert "Lesson 7 is the last lesson" in text
    assert "at most 0.5" not in words(html.replace(counts, ""))
    assert "homework" not in html.lower()


def test_a_copy_button_copies_its_own_prompt_and_says_so() -> None:
    html = PAGE.read_text()
    assert html.count('<button type="button">Copy</button>') == 6
    assert html.count('role="status"') == 1
    script = html[html.rindex("<script>") :]
    assert "navigator.clipboard.writeText(pre.textContent)" in script
    assert "selectNodeContents(pre)" in script
