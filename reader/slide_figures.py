# /// script
# requires-python = ">=3.11"
# dependencies = ["playwright", "pillow"]
# ///
"""The figures of the reader, cut from the published lesson decks and the class page.

A deck slide is rendered in the system Chrome at 1920 x 1080 with every build step shown,
and the figure is the slide's body without its title, kicker and page number. A class page
figure is one plotly chart of `artefacts/class-analysis/lesson<N>/index.html`, with the
student names taken off its axes.

    uv run reader/slide_figures.py
"""

import base64
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DECKS = ROOT / "presentations"
OUT = Path(__file__).resolve().parent / "img"

# Slide ids per deck; the figure is saved as img/<lesson>/<id>.png.
SLIDES = {
    "lesson1": ("lesson1-tensors.html", ["supervised", "choices"]),
    "lesson2": (
        "lesson2-kernels_states_gates.html",
        [
            "map",
            "beatfold",
            "axes",
            "kernel1d",
            "kernel2d",
            "down",
            "field",
            "fieldpool",
            "seqs",
            "diver",
            "walk",
            "depth-spread",
            "depth-gradient",
            "skiproute",
            "skip",
            "gate",
            "statescan",
            "scangate",
            "stategate",
            "catproblem",
            "squeeze",
            "family",
            "classes",
            "s-rhythm",
            "s-zoom",
            "s-crop",
            "s-question",
            "decomp",
            "bets",
            "design",
            "metric",
            "spine",
            "spine2",
        ],
    ),
}

# Interactive slides get a few presses of their forward or play button first, so the
# figure shows the instrument at work instead of its empty start.
PRESSES = {
    "kernel1d": [("#k1fwd", 9)],
    "kernel2d": [("#kfwd", 48)],
    "field": [("button:has-text('output → input')", 1), ("#rfaend", 1)],
    "fieldpool": [("button:has-text('output → input')", 1), ("#rfbend", 1)],
    "gate": [("#gtoggle", 1)],
}

HIDE = ".strip, .kick, .pn, .mk, h2.t, .ctl, .tp, button, .seg"


def deck_figures(page: Page, lesson: str, deck: str, ids: list[str]) -> None:
    url = (DECKS / deck).as_uri()
    page.goto(url)
    page.wait_for_timeout(800)
    numbers = page.evaluate(
        "() => Object.fromEntries([...document.querySelectorAll('.deck > section')]"
        ".map((s, i) => [s.id, i + 1]))"
    )
    (OUT / lesson).mkdir(parents=True, exist_ok=True)
    for sid in ids:
        page.goto(f"{url}#{numbers[sid]}")
        page.reload()
        page.wait_for_timeout(900)
        for button, n in PRESSES.get(sid, []):
            target = page.locator(f"section#{sid} {button}").first
            for _ in range(n):
                target.click()
                page.wait_for_timeout(150)
        page.evaluate(
            "() => ['crumbs', 'hint', 'notes'].forEach(i => document.getElementById(i)?.remove())"
        )
        # Every build step shown, as on the last press of the slide.
        page.evaluate(
            "() => document.querySelectorAll('[data-build-in]')"
            ".forEach(e => e.classList.add('built'))"
        )
        page.wait_for_timeout(700)
        section = page.locator(f"section#{sid}")
        section.evaluate(
            f"(s) => s.querySelectorAll('{HIDE}').forEach(e => e.style.visibility = 'hidden')"
        )
        body = section.locator(".body").first
        body.screenshot(path=str(OUT / lesson / f"{sid}.png"))
        print(lesson, sid)


# Class page tabs: (tab label, theme label or None, file name, plotly relayout).
CLASS_FIGURES = [
    ("What they investigated", "The encoder can move", "class_encoders", {}),
    ("What they investigated", "Capacity of the head", "class_capacity", {}),
    ("What they investigated", "Overfitting", "class_overfitting", {}),
    ("What they investigated", "A more complex head", "class_complexity", {}),
    ("Run again", None, "class_run_again", {"yaxis.showticklabels": False}),
    ("Search space", None, "class_search_space", {}),
    ("Two choices", None, "class_two_choices", {}),
    ("Cost", None, "class_cost", {}),
]


def class_figures(page: Page, lesson: int) -> None:
    page.set_viewport_size({"width": 1500, "height": 1300})
    page.goto(
        (
            ROOT / "artefacts" / "class-analysis" / f"lesson{lesson}" / "index.html"
        ).as_uri()
    )
    page.wait_for_timeout(2500)
    out = OUT / f"lesson{lesson}"
    out.mkdir(parents=True, exist_ok=True)
    for tab, theme, name, relayout in CLASS_FIGURES:
        page.locator("button.tab", has_text=tab).first.click()
        page.wait_for_timeout(1200)
        if theme:
            page.locator("button.theme-tab:visible", has_text=theme).first.click()
            page.wait_for_timeout(1500)
        plots = page.locator(".js-plotly-plot:visible")
        if plots.count() > 1:
            # A grid of small charts: the panel that holds them, as shown.
            panel = plots.first.locator("xpath=..")
            panel.evaluate(
                "(el) => { for (let a = el; a; a = a.parentElement) {"
                " a.style.overflow = 'visible'; a.style.maxHeight = 'none'; a.style.height = 'auto'; } }"
            )
            panel.screenshot(path=str(out / f"{name}.png"))
        else:
            # One chart, drawn whole by plotly at print size.
            data_url = plots.first.evaluate(
                "async (el, r) => { await Plotly.relayout(el, Object.assign("
                "{paper_bgcolor: '#f4f2ee', plot_bgcolor: '#f4f2ee'}, r));"
                " return Plotly.toImage(el, {format: 'png', width: 1300, height: 760, scale: 2}); }",
                relayout,
            )
            (out / f"{name}.png").write_bytes(
                base64.b64decode(data_url.split(",", 1)[1])
            )
        print("class", name, plots.count())
    # One grid of the two-choices chart on its own: dinov2-large, dropout against FLOPs, the
    # bottom-right panel of five by two.
    grid = Image.open(out / "class_two_choices.png")
    w, h = grid.size
    grid.crop((int(w * 0.795), int(h * 0.5), w, h)).save(
        out / "class_dropout_flops.png"
    )


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        for lesson, (deck, ids) in SLIDES.items():
            deck_figures(page, lesson, deck, ids)
        class_figures(page, 1)
        browser.close()


if __name__ == "__main__":
    main()
