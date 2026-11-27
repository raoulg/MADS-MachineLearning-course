"""Builds pages/presentaties/les3.html.

The frame (styles, navigation) is the one of the lesson 2 deck. Slides Raoul approved as
mockups are taken over as they are; the others are drawn here.
"""

import itertools
import json
import math
import random
import re
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
PRES = ROOT / "pages"
DECKS = PRES / "presentaties"
RESULTS = ROOT / "notebooks" / "3_scaling_and_search" / "results"

P, PT, V, VT, MUT, INK, RULE, BG = (
    "#1d5a73",
    "#c9dde4",
    "#c43a14",
    "#f3cdbf",
    "#5a544d",
    "#111111",
    "#cfc7bb",
    "#f4f1ec",
)
GREY, GREYT = "#8a837a", "#e4ded5"
SANS = "Outfit,Arial,sans-serif"
MONO = "'JetBrains Mono',monospace"


# ---------------------------------------------------------------- svg helpers
def txt(x, y, s, size=28, fill=MUT, anchor="start", weight=400, mono=False):
    fam = MONO if mono else SANS
    return (
        f'<text x="{x:.0f}" y="{y:.0f}" font-family="{fam}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>'
    )


def line(x1, y1, x2, y2, c=MUT, w=2.5, dash="", marker=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#m{marker})"' if marker else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" '
        f'stroke-width="{w}"{d}{m}/>'
    )


def rect(x, y, w, h, fill, stroke="none", sw=2.5, rx=0, dash="", op=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' fill-opacity="{op:.2f}"' if op < 1 else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
        f'fill="{fill}"{o} stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )


def circ(x, y, r, fill, stroke="none", sw=3):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def poly(pts, c, w=4, dash="", fill="none", op=1.0):
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    da = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' opacity="{op}"' if op < 1 else ""
    return f'<path d="{d}" fill="{fill}" stroke="{c}" stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round"{da}{o}/>'


def build(s):
    return f"<g data-build-in>{s}</g>"


def ribbon(x0, y0, x1, y1, w, fill, op=1.0):
    """A band of thickness w whose centre runs from (x0, y0) to (x1, y1)."""
    if w <= 0:
        return ""
    mx, h = (x0 + x1) / 2, w / 2
    d = (
        f"M {x0} {y0 - h} C {mx} {y0 - h} {mx} {y1 - h} {x1} {y1 - h} "
        f"L {x1} {y1 + h} C {mx} {y1 + h} {mx} {y0 + h} {x0} {y0 + h} Z"
    )
    return f'<path d="{d}" fill="{fill}" fill-opacity="{op}"/>'


MARKERS = "".join(
    f'<marker id="m{k}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" '
    f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{c}"/></marker>'
    for k, c in (("ink", INK), ("v", V), ("p", P), ("g", MUT))
)

# ---------------------------------------------------------------- slide frames
PART = {
    1: ("01 — scale", "notebook 3.1"),
    2: ("02 — mixture of experts", "notebook 3.2"),
    3: ("03 — hyperparameter optimization", "notebook 3.3"),
    4: ("04 — pareto fronts", "exercise 3.5"),
    5: ("your study", "exercise 3.5"),
}


def kick(part, goals):
    name, nb = PART[part]
    links = "".join(
        f'<a href="../leerdoelen/les3.html#g3-{g.split(".")[1]}">{g}</a>' for g in goals
    )
    goal = f"<p>goal {links}</p>" if goals else ""
    return f'<div class="kick"><p>■ {name}</p>{goal}<p>{nb}</p></div>'


def slide(sid, part, goals, title, picture, sentence="", credit="", mk="", h=560):
    c = f'\n    <p class="credit">{credit}</p>' if credit else ""
    s = f'\n    <p class="s">{sentence}</p>' if sentence else ""
    stamp = f'\n  <p class="mk">{mk}</p>' if mk else ""
    if picture.startswith(("<svg", "<div", "<img")):
        pic = picture
    else:
        pic = (
            f'<svg viewBox="0 0 1720 {h}" style="width:1720px; height:{h}px; flex-shrink:0" '
            f'role="img" aria-label="{title}">{picture}</svg>'
        )
    return f'''<section id="{sid}" class="paper std">
  <img class="strip" src="../img/lesson1/band_strip_paper.png" alt="pttrn logo in a Rule 30 strip">{stamp}
  {kick(part, goals)}
  <div class="body" style="gap:20px">
    <h2 class="t">{title}</h2>
    {pic}{s}{c}
  </div>
  <p class="pn n"></p>
</section>

'''


def divider(sid, part, crumb, claim, cards):
    _, nb = PART[part]
    cs = "".join(
        f'<div class="card"><p class="ck">{i} · {k}</p><p class="s">{s}</p></div>'
        for i, (k, s) in enumerate(cards, 1)
    )
    return f'''<section id="{sid}" class="ink std" data-crumb="{crumb}">
  <img class="strip" src="../img/lesson1/band_strip_ink.png" alt="pttrn logo in a Rule 30 strip">
  <div class="kick"><p>▲ part 0{part}</p><p>{nb}</p></div>
  <div style="display:flex; flex-direction:column; gap:64px; justify-content:center; flex:1">
    <h2 class="claim">{claim}</h2>
    <div class="cards">{cs}</div>
  </div>
  <p class="pn n"></p>
</section>

'''


def taken(path, sid, part, goals, title=None, mk=""):
    """A section of an approved mockup, with this deck's strip and kick."""
    src = (HERE / path).read_text()
    found = re.search(rf'<section id="{sid}".*?</section>', src, re.DOTALL)
    if found is None:
        raise ValueError(f"{path} holds no section {sid}")
    sec = found.group(0)
    sec = re.sub(
        r'<section id="([^"]+)"[^>]*>',
        r'<section id="\1" class="paper std">',
        sec,
        count=1,
    )
    sec = re.sub(
        r'src="[^"]*band_strip_paper.png"',
        'src="../img/lesson1/band_strip_paper.png"',
        sec,
    )
    sec = re.sub(
        r'<div class="kick">.*?</div>',
        lambda m: kick(part, goals),
        sec,
        count=1,
        flags=re.DOTALL,
    )
    if title:
        sec = re.sub(
            r'<h2 class="t">.*?</h2>',
            lambda m: f'<h2 class="t">{title}</h2>',
            sec,
            count=1,
            flags=re.DOTALL,
        )
    if mk:
        sec = sec.replace(
            '<div class="kick">', f'<p class="mk">{mk}</p>\n  <div class="kick">', 1
        )
    sec = sec.replace('<p class="s" data-build-in>', '<p class="s">')
    return sec + "\n\n"


# ---------------------------------------------------------------- cover and map


# ---------------------------------------------------------------- part 1


def law_price():
    rows = [
        ("1M → 10M parameters", 8, "+9M"),
        ("10M → 100M", 78, "+90M"),
        ("100M → 1B", 780, "+900M"),
    ]
    s = txt(520, 34, "what it buys", 28, P, weight=600) + txt(
        800, 34, "what it costs", 28, V, weight=600
    )
    for i, (name, cost, lab) in enumerate(rows):
        y = 70 + i * 135
        g = txt(480, y + 58, name, 32, INK, "end", 500)
        g += rect(520, y, 220, 90, P) + txt(
            630, y + 56, "loss × 0.84", 30, "#ffffff", "middle", 500
        )
        g += rect(800, y, cost, 90, V) + txt(
            800 + cost + 18, y + 56, lab, 32, V, weight=600
        )
        s += g if i == 0 else build(g)
    return slide(
        "law-price",
        1,
        ["3.2"],
        "What the next 10× buys, and what it costs",
        s,
        "The gain is the same factor every time. The price is ten times the previous one.",
        h=470,
    )


# ---------------------------------------------------------------- part 2
def family():
    src = (DECKS / "les2.html").read_text()
    found = re.search(r'<section id="family".*?</section>', src, re.DOTALL)
    if found is None:
        raise ValueError("the lesson 2 deck holds no section family")
    sec = found.group(0)
    sec = sec.replace('<section id="family"', '<section id="routes-family"')
    sec = re.sub(
        r'<div class="kick">.*?</div>',
        lambda m: kick(2, ["3.4"]),
        sec,
        count=1,
        flags=re.DOTALL,
    )
    sec = re.sub(
        r'<div class="callout">.*?</div>',
        '<div class="callout"><p class="s">Lesson 2 ended on this picture. Today: the highway, and the mixture of experts.</p></div>',
        sec,
        count=1,
        flags=re.DOTALL,
    )
    sec = re.sub(r"<aside>.*?</aside>", "", sec, flags=re.DOTALL)
    sec = sec.replace('<p class="s" data-build-in>', '<p class="s">')
    return sec + "\n\n"


ROUTING = {
    "trained": {
        "Galaxy10": [0, 1.0, 0, 0, 0, 0, 0, 0],
        "BloodMNIST": [0, 0, 0, 0, 0, 1.0, 0, 0],
        "Food-101": [0.07, 0.07, 0.69, 0.07, 0.09, 0, 0, 0],
        "Flowers": [0, 0.11, 0, 0, 0, 0.05, 0.83, 0],
        "GTSRB": [0, 0, 0, 0, 0.33, 0.67, 0, 0],
        "DTD": [0.15, 0, 0.05, 0, 0, 0, 0.19, 0.60],
        "EuroSAT": [0.25, 0, 0, 0, 0, 0, 0.17, 0.58],
    },
    "untrained": {
        "Galaxy10": [0, 0.23, 0.06, 0.12, 0, 0.10, 0.25, 0.21],
        "BloodMNIST": [0.14, 0.16, 0, 0, 0.07, 0.33, 0, 0.26],
        "Food-101": [0.12, 0.08, 0.22, 0.08, 0.16, 0.14, 0.07, 0.13],
        "Flowers": [0.11, 0.19, 0.08, 0.07, 0, 0.27, 0, 0.24],
        "GTSRB": [0.19, 0.07, 0.06, 0, 0.18, 0.20, 0.15, 0.12],
        "DTD": [0.06, 0.11, 0.05, 0.11, 0, 0.26, 0, 0.33],
        "EuroSAT": [0.36, 0, 0.08, 0, 0.14, 0.07, 0.12, 0.19],
    },
}


# ---------------------------------------------------------------- part 3


# ---------------------------------------------------------------- part 4: the admissions detour


# ---------------------------------------------------------------- part 5


# ---------------------------------------------------------------- assemble

OK = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#D55E00",
    "#0072B2",
]  # Okabe-Ito, as notebook 3.1


def mixc(a, b, t):
    pa = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    pb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(
        f"{round(x + (y - x) * t):02x}" for x, y in zip(pa, pb, strict=True)
    )


def vtxt(x, y, s, size=26, fill=MUT):
    return (
        f'<text x="{x:.0f}" y="{y:.0f}" font-family="{SANS}" font-size="{size}" '
        f'fill="{fill}" text-anchor="middle" transform="rotate(-90 {x:.0f} {y:.0f})">{s}</text>'
    )


# ---------------------------------------------------------------- the measured curves
def scaling_runs():
    rows = json.loads((RESULTS / "food_scaling.json").read_text())
    return [r for r in rows if r["width"] == 0]


def scaling_curves(x0, y0, w, h, lw=5, dot=11):
    """The test loss of notebook 3.1's five runs against optimizer steps, both log.
    Returns the drawing, and per run the end of its curve and its colour."""
    X = lambda s: x0 + w * (math.log10(s) - 1) / (math.log10(3000) - 1)
    Y = lambda v: (
        y0 - h * (math.log(v) - math.log(0.2)) / (math.log(1.0) - math.log(0.2))
    )
    g, ends = "", []
    for run, col in zip(scaling_runs(), OK, strict=True):
        c = run["curve"]
        pts = [
            (X(s), Y(v))
            for s, v in zip(c["step"], c["held"], strict=True)
            if s >= 10 and v <= 1.0
        ]
        g += poly(pts, col, lw)
        g += circ(X(run["best_step"]), Y(run["held"]), dot, col)
        ends.append((pts[-1], col, run["n"]))
    return g, ends, X, Y


# ---------------------------------------------------------------- cover and map
def glyph(kind):
    if kind == 1:
        g = scaling_curves(0, 128, 260, 150, lw=4, dot=6)[0]
    elif kind == 2:
        g = rect(0, 45, 26, 40, "#86b9c9")
        for y, op in ((6, 0.2), (45, 1.0), (84, 0.35)):
            g += line(30, 65, 96, y + 20, "#5a544d", 2)
            g += "".join(
                rect(100 + i * 28, y, 26, 40, "#f2643c", op=op) for i in range(5)
            )
    elif kind == 3:
        g = (
            '<ellipse cx="150" cy="62" rx="46" ry="34" fill="#86b9c9" fill-opacity="0.3"/>'
            + "".join(
                circ(x, y, 7, "#b8b0a5")
                for x, y in (
                    (20, 30),
                    (52, 96),
                    (88, 24),
                    (232, 22),
                    (214, 104),
                    (30, 70),
                )
            )
            + "".join(
                circ(x, y, 7, "#f2643c")
                for x, y in ((128, 52), (150, 78), (166, 48), (146, 60))
            )
        )
    else:
        fr = [(40, 14), (115, 30), (180, 60), (232, 104)]
        inside = ((48, 84), (92, 106), (124, 82), (152, 112), (76, 72))
        outside = ((70, 44), (140, 52), (204, 112), (216, 96))
        g = (
            rect(12, 60, 168, 62, "#86b9c9", op=0.22)
            + line(12, 4, 12, 122, "#8a837a", 2)
            + line(12, 122, 256, 122, "#8a837a", 2)
            + "".join(circ(x, y, 6, "#b8b0a5") for x, y in inside + outside)
            + poly(fr, "#f2643c", 5)
            + "".join(circ(x, y, 8, "#f2643c") for x, y in fr)
        )
    return f'<svg viewBox="-5 -5 270 140" style="width:270px; height:140px" aria-hidden="true">{g}</svg>'


def cover():
    art = scaling_curves(0, 330, 640, 380, lw=6, dot=12)[0]
    return f"""<section id="cover" class="ink" style="display:flex; flex-direction:column; padding:236px 100px 56px">
  <img src="../img/lesson1/band_title_ink.png" alt="pttrn logo in a Rule 30 band" style="position:absolute; left:0; top:0; width:1920px; height:201px; object-fit:contain">
  <svg viewBox="-20 -20 700 380" style="position:absolute; top:280px; right:110px; width:700px; height:380px" aria-hidden="true">{art}</svg>
  <div class="kick"><p style="color:#b8b0a5">MADS-ML</p><p style="color:#b8b0a5">Machine Learning</p></div>
  <div style="flex:1"></div>
  <div style="display:flex; flex-direction:column; gap:32px">
    <div style="width:280px; height:2px; background:#f2643c"></div>
    <h1 style="font-family:Outfit, Arial, sans-serif; font-size:132px; font-weight:500; line-height:0.98; letter-spacing:-4px; color:#f4f1ec">Scaling<br>and search</h1>
    <p style="max-width:1000px; font-family:Outfit, Arial, sans-serif; font-size:40px; line-height:1.3; color:#b8b0a5">Working within the limits of compute, data and time</p>
  </div>
  <div class="foot" style="border-top-color:#2b2621"><p style="color:#b8b0a5">Lesson 3</p><p style="color:#b8b0a5">Hogeschool Utrecht</p></div>
</section>

"""


def agenda():
    parts = [
        (1, "Scale", "what more parameters and data buy"),
        (2, "Mixture of experts", "many parameters stored, few used"),
        (3, "Hyperparameter optimization", "an algorithm chooses the next trial"),
        (4, "Pareto fronts", "which model is best"),
    ]
    cells = "".join(
        f'<div class="top{" first" if n == 1 else ""}" style="gap:18px"><p class="ck2">0{n}</p>'
        f'<div style="background:#141210; padding:14px 20px">{glyph(n)}</div>'
        f'<p class="s" style="font-size:34px; font-weight:500">{name}</p><p class="m" style="font-size:26px">{q}</p></div>'
        for n, name, q in parts
    )
    return f"""<section id="map" class="paper std" data-crumb="agenda">
  <img class="strip" src="../img/lesson1/band_strip_paper.png" alt="pttrn logo in a Rule 30 strip">
  <div class="kick"><p>Agenda</p><p>four parts</p><p><a href="../leerdoelen/les3.html">all goals</a></p></div>
  <div class="body" style="gap:60px">
    <h2 class="t">Today: working within the limits of <span class="v">compute, data and time</span></h2>
    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:44px">{cells}</div>
  </div>
  <p class="pn n"></p>
</section>

"""


# ---------------------------------------------------------------- part 1
def claim_or_size():
    x0, full, bh = 430, 800, 76
    w = lambda k: full * k / 120
    row = lambda y, name: txt(x0 - 30, y + 50, name, 34, INK, "end", 500)
    gate = lambda x, y: (
        rect(x - 130, y + 12, 118, bh - 24, V)
        + txt(x - 71, y + 50, "gate", 28, "#ffffff", "middle", 600)
    )
    note = lambda y, s, col: txt(x0 + w(120) + 40, y + 50, s, 34, col, weight=600)
    s = (
        row(30, "baseline")
        + rect(x0, 30, w(75), bh, P)
        + txt(x0 + 20, 80, "75k parameters", 30, "#ffffff", weight=500)
    )
    two = (
        row(190, "with gate")
        + rect(x0, 190, w(75), bh, P)
        + rect(x0 + w(75), 190, w(45), bh, mixc(P, BG, 0.45))
        + txt(x0 + 20, 240, "120k parameters", 30, "#ffffff", weight=500)
        + gate(x0 + w(120), 190)
        + note(190, "changed two things", V)
    )
    one = (
        row(350, "with gate")
        + rect(x0, 350, w(75), bh, P)
        + txt(x0 + 20, 400, "75k parameters", 30, "#ffffff", weight=500)
        + gate(x0 + w(75), 350)
        + note(350, "changed one thing", P)
        + line(x0 + w(75), 20, x0 + w(75), 440, P, 3, "10 8")
    )
    return slide(
        "claim-or-size",
        1,
        ["3.1"],
        'Change <span class="v">one thing</span> to see the impact',
        s + build(two) + build(one),
        "A model with a gate and more parameters differs from the baseline in two ways. With the baseline's parameters, the gate is the one thing that changed.",
        h=470,
    )


def law_curves():
    x0, y0, w, h = 330, 470, 1000, 430
    g, ends, X, Y = scaling_curves(x0, y0, w, h)
    s = line(x0, y0, x0 + w + 40, y0, MUT, 2.5, marker="g") + line(
        x0, y0, x0, y0 - h - 30, MUT, 2.5, marker="g"
    )
    s += txt(x0 + w + 40, y0 + 88, "optimizer steps (log) →", 28, MUT, "end") + txt(
        x0 + 20, y0 - h - 6, "test loss (log) ↑", 28, MUT
    )
    for v in (10, 100, 1000):
        s += line(X(v), y0, X(v), y0 + 10, MUT, 2) + txt(
            X(v), y0 + 44, f"{v:,}", 26, MUT, "middle"
        )
    for v in (0.2, 0.3, 0.5, 1.0):
        s += line(x0 - 10, Y(v), x0, Y(v), MUT, 2) + txt(
            x0 - 24, Y(v) + 9, f"{v:g}", 26, MUT, "end"
        )
    for (x, y), col, n in ends:
        s += txt(x + 18, y + 9, f"{n:,} images", 28, col, weight=600)
    return slide(
        "law-curves",
        1,
        ["3.2", "3.10"],
        'Our own runs: more images, a <span class="v">lower test loss</span>',
        s + g,
        "One model, five amounts of data: every run ends below the one before it. The dot marks where a run's validation loss was lowest.",
        "Food-101, a linear head on a frozen encoder · notebook 3.1.3",
        h=570,
    )


def grow_together():
    cw, ch, gx, gy = 600, 220, 40, 30
    x0, y0 = 330, 20
    cell = lambda c, r: (x0 + c * (cw + gx), y0 + r * (ch + gy))

    def frame(c, r, fill, label, col):
        x, y = cell(c, r)
        return rect(x, y, cw, ch, fill, RULE, 2) + txt(
            x + cw - 20, y + 40, label, 28, col, "end", 600
        )

    def curve(c, r, f, col, wd=5):
        x, y = cell(c, r)
        return poly(
            [
                (x + 30 + (cw - 60) * i / 40, y + ch - 24 - (ch - 70) * f(i / 40))
                for i in range(41)
            ],
            col,
            wd,
        )

    fall = lambda top, floor, k: lambda u: floor + (top - floor) * math.exp(-k * u)
    s = frame(0, 1, BG, "few parameters, little data", MUT)
    s += curve(0, 1, fall(0.95, 0.5, 4), GREY, 3) + curve(0, 1, fall(0.97, 0.56, 4), P)
    a = frame(0, 0, mixc(VT, BG, 0.4), "overfits", V)
    a += curve(0, 0, fall(0.95, 0.12, 4), GREY, 3)
    a += curve(0, 0, lambda u: 0.42 + 0.53 * math.exp(-7 * u) + 0.26 * u * u, P)
    b = frame(1, 1, mixc(GREYT, BG, 0.3), "stops improving", MUT)
    b += curve(1, 1, fall(0.95, 0.46, 9), GREY, 3) + curve(1, 1, fall(0.96, 0.47, 9), P)
    c = frame(1, 0, mixc(PT, BG, 0.4), "every step lies lower", P)
    x, y = cell(1, 0)
    for i, col in enumerate(OK[1:]):
        u0, u1 = 0.1 * i, 0.52 + 0.16 * i
        top, floor = 0.95 - 0.1 * i, 0.62 - 0.15 * i
        c += poly(
            [
                (
                    x + 30 + (cw - 60) * u,
                    y
                    + ch
                    - 24
                    - (ch - 70) * (floor + (top - floor) * math.exp(-6 * (u - u0))),
                )
                for u in [u0 + (u1 - u0) * j / 40 for j in range(41)]
            ],
            col,
            5,
        )
    lab = txt(x0 - 30, y0 + ch / 2 + 10, "more parameters", 30, INK, "end", 500) + txt(
        x0 - 30, y0 + ch + gy + ch / 2 + 10, "few parameters", 30, INK, "end", 500
    )
    lab += txt(
        x0 + cw / 2, y0 + 2 * ch + gy + 46, "little data", 30, INK, "middle", 500
    )
    lab += txt(
        x0 + cw + gx + cw / 2,
        y0 + 2 * ch + gy + 46,
        "more data",
        30,
        INK,
        "middle",
        500,
    )
    leg = line(60, 540, 110, 540, GREY, 3) + txt(122, 549, "train loss", 24, MUT)
    leg += line(60, 580, 110, 580, P, 5) + txt(122, 589, "validation loss", 24, MUT)
    return slide(
        "grow-together",
        1,
        ["3.3"],
        'Parameters and data <span class="v">grow together</span>',
        s + lab + leg + build(a) + build(b) + build(c),
        "More parameters on the same data overfit; more data for the same model stops paying. Grown together, every step ends lower.",
        mk="schematic",
        h=600,
    )


def gopher():
    rows = [("417M", 25.7), ("1.4B", 27.3), ("7.1B", 29.5), ("280B", 60.0)]
    x0, y0, sc = 420, 470, 6.2
    s = line(x0 - 20, y0, 1500, y0, MUT, 2.5)
    s += txt(x0 - 40, 60, "knowledge test score ↑", 28, MUT)
    for v in (0, 25, 50):
        s += txt(x0 - 40, y0 - v * sc + 9, f"{v}%", 26, MUT, "end")
    s += line(x0 - 20, y0 - 25 * sc, 1500, y0 - 25 * sc, MUT, 2, "10 8") + txt(
        1516, y0 - 25 * sc + 9, "guessing: 1 in 4", 26, MUT
    )
    for i, (name, v) in enumerate(rows):
        x = x0 + 40 + i * 260
        big = i == 3
        s += rect(x, y0 - v * sc, 170, v * sc, P)
        s += txt(x + 85, y0 - v * sc - 16, f"{v:.0f}%", 32, P, "middle", 600)
        s += txt(x + 85, y0 + 46, name, 30, INK, "middle", 500)
        if big:
            s += txt(x + 85, y0 - v * sc + 46, "Gopher", 30, "#ffffff", "middle", 600)
    s += txt(x0 - 40, y0 + 46, "parameters", 30, INK, "end", 500)
    return slide(
        "gopher",
        1,
        ["3.3"],
        'More parameters on the <span class="v">same data</span>',
        s,
        "DeepMind trained one model in several sizes, all on the same 300B tokens. The largest, Gopher, was far better than the rest.",
        "Rae et al. (2021), Scaling language models: methods, analysis &amp; insights from training Gopher · MMLU, Table A8.",
        h=540,
    )


def compute_area():
    x0, y0 = 330, 500
    sx, sy = 3.6, 0.31
    s = line(x0, y0, x0 + 1120, y0, MUT, 2.5, marker="g") + line(
        x0, y0, x0, 20, MUT, 2.5, marker="g"
    )
    s += txt(x0 + 1120, y0 + 86, "parameters →", 28, MUT, "end") + txt(
        x0 - 20, 40, "training tokens ↑", 28, MUT, "end"
    )
    for v, lab in ((70, "70B"), (280, "280B")):
        s += line(x0 + v * sx, y0, x0 + v * sx, y0 + 10, MUT, 2) + txt(
            x0 + v * sx, y0 + 44, lab, 26, MUT, "middle"
        )
    for v, lab in ((300, "300B"), (1400, "1.4T")):
        s += line(x0 - 10, y0 - v * sy, x0, y0 - v * sy, MUT, 2) + txt(
            x0 - 20, y0 - v * sy + 9, lab, 26, MUT, "end"
        )
    s += rect(x0, y0 - 300 * sy, 280 * sx, 300 * sy, PT, P, 4, op=0.85)
    s += txt(x0 + 280 * sx - 24, y0 - 300 * sy / 2 + 12, "Gopher", 34, P, "end", 600)
    b = rect(x0, y0 - 1400 * sy, 70 * sx, 1400 * sy, VT, V, 4, op=0.75)
    b += txt(x0 + 70 * sx + 24, y0 - 1400 * sy + 44, "Chinchilla", 34, V, weight=600)
    return slide(
        "compute-area",
        1,
        ["3.3"],
        'Both models used <span class="v">5.76 × 10²³ FLOPs</span>',
        s + build(b),
        "",
        "Rae et al. (2021), Gopher · Hoffmann et al. (2022), Chinchilla; FLOPs of Gopher's training: their Table 3.",
        h=590,
    )


def bet_or_scale():
    x0, y0 = 300, 500
    s = line(x0, y0, 1560, y0, MUT, 2.5, marker="g") + line(
        x0, y0, x0, 30, MUT, 2.5, marker="g"
    )
    s += txt(1560, y0 + 50, "parameters (log) →", 28, MUT, "end") + txt(
        x0 - 20, 50, "loss (log) ↑", 28, MUT, "end"
    )
    ya = lambda x: 110 + (x - 360) * 0.175
    yb = lambda x: ya(x) + 105
    s += poly([(360, ya(360)), (1440, ya(1440))], P, 6) + txt(
        1460, ya(1440) + 10, "architecture A", 30, P, weight=600
    )
    b = poly([(360, yb(360)), (1440, yb(1440))], V, 6) + txt(
        1460, yb(1440) + 10, "architecture B", 30, V, weight=600
    )
    b += line(900, 40, 900, y0, MUT, 2.5, "10 8")
    b += circ(900, ya(900), 13, BG, P, 5) + circ(900, yb(900), 13, VT, V, 5)
    return slide(
        "bet-or-scale",
        1,
        ["3.1", "3.11"],
        'Compare architectures with <span class="v">equal parameters and data</span>',
        s + build(b),
        mk="schematic",
        h=560,
    )


# ---------------------------------------------------------------- part 2
def vec(x, y, col, op=1.0, n=5, cw=34, ch=46, dash=False):
    """A vector as a row of cells, centred on y; its colour strength is its weight."""
    d = ' stroke-dasharray="6 5"' if dash else ""
    return "".join(
        f'<rect x="{x + i * cw:.0f}" y="{y - ch / 2:.0f}" width="{cw - 4}" height="{ch}" '
        f'fill="{col}" fill-opacity="{op:.2f}" stroke="{GREY if dash else col}" stroke-width="2"{d}/>'
        for i in range(n)
    )


def opbox(x, y, label, w=150, on=True):
    col = V if on else GREY
    d = "" if on else "8 6"
    return rect(x, y - 30, w, 60, BG, col, 3, dash=d) + txt(
        x + w / 2, y + 10, label, 26, col, "middle", 600
    )


def highway_g():
    def panel(x0, g, title):
        cy, top, bot = 290, 170, 410
        s = txt(x0, 40, title, 30, INK, weight=500)
        s += vec(x0, cy, P) + txt(x0 + 83, cy + 62, "x", 28, INK, "middle", 600)
        s += line(x0 + 176, cy - 10, x0 + 236, top, MUT, 2.5, marker="g")
        s += poly([(x0 + 176, cy + 10), (x0 + 236, bot), (x0 + 424, bot)], MUT, 2.5)
        s += line(x0 + 420, bot, x0 + 436, bot, MUT, 2.5, marker="g")
        s += txt(x0 + 330, bot - 16, "skip", 26, P, "middle", 600)
        s += opbox(x0 + 246, top, "block")
        s += line(x0 + 400, top, x0 + 436, top, MUT, 2.5, marker="g")
        s += vec(x0 + 446, top, V, g)
        s += txt(x0 + 531, top - 44, f"⊗ g = {g}", 26, V, "middle", mono=True)
        s += vec(x0 + 446, bot, P, 1 - g)
        s += txt(
            x0 + 531, bot + 62, f"⊗ (1 − g) = {1 - g:.1f}", 26, P, "middle", mono=True
        )
        s += line(x0 + 622, top, x0 + 660, cy - 14, MUT, 2.5, marker="g")
        s += line(x0 + 622, bot, x0 + 660, cy + 14, MUT, 2.5, marker="g")
        s += circ(x0 + 684, cy, 22, BG, MUT, 2.5) + txt(
            x0 + 684, cy + 11, "+", 32, MUT, "middle", 600
        )
        s += vec(x0 + 722, cy, mixc(P, V, g), n=3)
        s += txt(x0 + 771, cy + 62, "y", 28, INK, "middle", 600)
        return s

    return slide(
        "highway-g",
        2,
        ["3.4"],
        'A highway: a gate <span class="v">weighs</span> two routes',
        panel(40, 0.8, "one input: the gate is mostly open")
        + build(panel(900, 0.2, "another input: mostly closed")),
        "Both routes read the whole vector. The gate scales the block's output by g and the skip by 1 − g, and the two are added.",
        h=500,
    )


def highway_to_moe():
    ys = (90, 200, 310, 420)
    cols = [P, V, mixc(V, P, 0.2), mixc(P, V, 0.2)]

    def blend(weights):
        rgb = [[int(c[i : i + 2], 16) for i in (1, 3, 5)] for c in cols]
        return "#" + "".join(
            f"{round(sum(w * c[k] for w, c in zip(weights, rgb, strict=True))):02x}"
            for k in range(3)
        )

    def panel(x0, weights, title, sparse):
        cy = 255
        s = txt(x0, 30, title, 30, INK, weight=500)
        s += vec(x0, cy, P) + txt(x0 + 83, cy + 62, "x", 28, INK, "middle", 600)
        top = max(range(4), key=lambda i: weights[i])
        used = [float(i == top) for i in range(4)] if sparse else weights
        # the router is the fork: it gives every route its weight
        s += line(x0 + 172, cy, x0 + 190, cy, MUT, 2.5, marker="g")
        for i, y in enumerate(ys):
            runs = used[i] > 0
            if runs:
                s += poly([(x0 + 300, cy), (x0 + 346, y)], cols[i], 2 + 9 * used[i])
            else:
                s += poly([(x0 + 300, cy), (x0 + 346, y)], RULE, 2.5, "6 6")
            s += opbox(x0 + 350, y, f"expert {i + 1}", w=124, on=runs)
            if not runs:
                s += vec(x0 + 492, y, BG, 0, cw=28, dash=True)
                s += txt(x0 + 562, y - 34, "not run", 24, GREY, "middle", mono=True)
                continue
            s += line(x0 + 474, y, x0 + 486, y, MUT, 2.5, marker="g")
            s += vec(x0 + 492, y, cols[i], min(1.0, 0.25 + 1.5 * used[i]), cw=28)
            s += txt(x0 + 562, y - 34, f"{used[i]:g}", 24, cols[i], "middle", mono=True)
            s += line(x0 + 636, y, x0 + 662, cy + (y - cy) * 0.12, MUT, 2.5, marker="g")
        s += rect(x0 + 196, cy - 34, 104, 68, BG, INK, 3) + txt(
            x0 + 248, cy + 9, "router", 26, INK, "middle", 600
        )
        s += circ(x0 + 686, cy, 22, BG, MUT, 2.5) + txt(
            x0 + 686, cy + 11, "+", 32, MUT, "middle", 600
        )
        s += vec(x0 + 716, cy, blend(used), n=3, cw=28) + txt(
            x0 + 758, cy + 62, "y", 28, INK, "middle", 600
        )
        return s

    w = [0.15, 0.45, 0.30, 0.10]
    return slide(
        "highway-to-moe",
        2,
        ["3.4", "3.5"],
        'A mixture of experts: a highway with <span class="v">more routes</span>',
        panel(20, w, "every expert, weighed", False)
        + build(panel(880, w, "sparse, top-1: only the highest weight runs", True)),
        "The router reads x and gives every route a weight; the weights sum to 1. Notebook 3.2 uses the sparse form: 8 experts, of which one runs per image.",
        h=490,
    )


def moe_split():
    cw, ch = 72, 56
    even, band, neutral = 1 / 8, 0.05, "#e4ded5"

    def colour(v):
        lo, hi = even - band, even + band
        if v < lo:
            return mixc(neutral, P, 0.25 + 0.6 * (lo - v) / lo)
        if v > hi:
            return mixc(neutral, V, 0.25 + 0.75 * ((v - hi) / (1 - hi)) ** 0.6)
        return neutral

    def grid(x0, which, title):
        g = txt(x0, 34, title, 30, INK, weight=500)
        for r, (dom, shares) in enumerate(ROUTING[which].items()):
            y = 60 + r * ch
            g += txt(x0 - 20, y + 38, dom, 26, MUT, "end")
            for c, v in enumerate(shares):
                g += rect(x0 + c * cw, y, cw - 4, ch - 4, colour(v))
        for c in range(8):
            g += txt(
                x0 + c * cw + cw / 2 - 2,
                60 + 7 * ch + 30,
                str(c),
                24,
                MUT,
                "middle",
                mono=True,
            )
        g += txt(x0 + 8 * cw + 16, 60 + 7 * ch + 30, "expert", 24, MUT)
        return g

    lx, ly, lw = 560, 530, 600
    leg = "".join(
        rect(lx + lw * i / 120, ly, lw / 120 + 1, 24, colour(i / 119))
        for i in range(120)
    )
    pos = lambda v: lx + lw * v
    for v, lab in (
        (0, "0%"),
        (even, "12.5%"),
        (0.5, "50%"),
        (1, "100%"),
    ):
        leg += line(pos(v), ly + 24, pos(v), ly + 34, MUT, 2) + txt(
            pos(v),
            ly + 62,
            lab,
            24,
            MUT,
            "end" if v in (0, 1) else "middle",
        )
    leg += txt(lx - 20, ly + 20, "share of a kind's images", 24, MUT, "end")
    leg += txt(pos(even - band), ly - 12, "an even share, ± 5%", 24, MUT)
    return slide(
        "moe-split",
        2,
        ["3.5"],
        'Where does the router <span class="v">send</span> each kind of image?',
        grid(230, "untrained", "before training")
        + leg
        + build(grid(1040, "trained", "after training")),
        "Nobody tells the router which kind of image it reads. After training, galaxies, flowers and dishes each have an expert of their own.",
        "Share of each domain's test images per expert · notebook 3.2.5",
        h=600,
    )


def moe_collapse():
    T = 60

    def rich():
        s = [0.150, 0.152, 0.240, 0.150, 0.156, 0.152]
        out = [s]
        for _ in range(T):
            p = [v**1.075 for v in s]
            s = [v / sum(p) for v in p]
            out.append(s)
        return out

    def even():
        out = []
        for t in range(T + 1):
            lead = 0.240 - 0.073 * (1 - math.exp(-t / 10))
            s = [1 / 6 + 0.012 * math.sin(0.25 * t + 1.7 * i) for i in range(6)]
            s[2] = lead
            out.append([v / sum(s) for v in s])
        return out

    def row(y0, series, name):
        x0, w, h = 380, 720, 200
        g = txt(350, y0 + h / 2 - 8, name[0], 30, INK, "end", 500) + txt(
            350, y0 + h / 2 + 30, name[1], 30, INK, "end", 500
        )
        cum = [[sum(s[:i]) for i in range(7)] for s in series]
        X = lambda t: x0 + w * t / T
        for i in range(6):
            up = [(X(t), y0 + h * cum[t][i]) for t in range(T + 1)]
            lo = [(X(t), y0 + h * cum[t][i + 1]) for t in reversed(range(T + 1))]
            g += poly(up + lo, BG, 3, fill=V if i == 2 else GREYT)
        g += line(x0 + w + 20, y0 + h / 2, x0 + w + 90, y0 + h / 2, MUT, 3, marker="g")
        bx = x0 + w + 130
        g += line(bx - 10, y0 + h, bx + 6 * 70, y0 + h, MUT, 2.5)
        for i, v in enumerate(series[-1]):
            g += rect(bx + i * 70, y0 + h - v * h, 54, v * h, V if i == 2 else GREY)
            g += txt(bx + i * 70 + 27, y0 + h + 30, f"E{i + 1}", 24, MUT, "middle")
        return g

    s = row(20, rich(), ("without a", "balancing loss")) + build(
        row(300, even(), ("with a", "balancing loss"))
    )
    s += txt(380, 270, "share of the images per expert, during training →", 24, MUT)
    return slide(
        "moe-collapse",
        2,
        ["3.5"],
        'Router collapse: <span class="v">one expert</span> gets everything',
        s,
        "The expert that happens to start best gets more images, learns faster from them, and gets even more. A balancing loss grows the more uneven the shares are, and so spreads the images again.",
        "Collapse: Shazeer et al. (2017) · balancing loss: Fedus, Zoph &amp; Shazeer (2022), Switch Transformers",
        mk="schematic",
        h=545,
    )


# ---------------------------------------------------------------- part 3
GOOD = (
    0.42,
    0.58,
    0.105,
    0.2,
)  # centre and radii of the region where trials score well


def inside(p):
    cx, cy, rx, ry = GOOD
    return ((p[0] - cx) / rx) ** 2 + ((p[1] - cy) / ry) ** 2 <= 1


def panel(x0, y0, w, h, title, groups, label=True):
    """groups: list of (points, kind); kind in grey, hollow, p, v."""
    cx, cy, rx, ry = GOOD
    g = txt(x0, y0 - 18, title, 30, INK, weight=500)
    g += (
        f'<ellipse cx="{x0 + w * cx:.0f}" cy="{y0 + h * (1 - cy):.0f}" rx="{w * rx:.0f}" '
        f'ry="{h * ry:.0f}" fill="{PT}" fill-opacity="0.7"/>'
    )
    if label:
        g += txt(x0 + w * cx, y0 + h * (1 - cy - ry) - 12, "good", 24, P, "middle", 600)
    g += rect(x0, y0, w, h, "none", MUT, 2.5)
    g += txt(x0 + w / 2, y0 + h + 40, "hyperparameter 1", 24, MUT, "middle")
    g += vtxt(x0 - 22, y0 + h / 2, "hyperparameter 2", 24)
    style = {"grey": (GREY, GREY), "hollow": (BG, GREY), "p": (P, P), "v": (V, V)}
    for pts, kind in groups:
        f, st = style[kind]
        g += "".join(circ(x0 + w * u, y0 + h * (1 - v), 12, f, st, 4) for u, v in pts)
    return g


def trials():
    rnd = random.Random(3)
    grid = [(0.125 + 0.25 * i, 0.125 + 0.25 * j) for i in range(4) for j in range(4)]
    rand = [
        (u + rnd.uniform(-0.095, 0.095), v + rnd.uniform(-0.095, 0.095))
        for u, v in grid
    ]
    first = [(0.1, 0.3), (0.24, 0.8), (0.4, 0.62), (0.63, 0.4), (0.8, 0.2), (0.9, 0.72)]
    tpe = [
        (0.47, 0.5), (0.56, 0.58), (0.52, 0.36), (0.44, 0.6), (0.5, 0.66),
        (0.55, 0.44), (0.48, 0.42), (0.585, 0.52), (0.43, 0.48), (0.51, 0.55),
    ]  # fmt: skip
    parents = [first[2], first[3]]
    gen2 = [(0.3, 0.5), (0.36, 0.78), (0.42, 0.44), (0.575, 0.6), (0.77, 0.22)]
    par2 = [gen2[2], gen2[3]]
    gen3 = [(0.52, 0.39), (0.46, 0.33), (0.5, 0.6), (0.545, 0.47), (0.53, 0.7)]
    near = lambda ps: [(u - 0.08, v + 0.08) for u, v in ps]
    tpe, gen2, gen3 = near(tpe), near(gen2), near(gen3)
    par2 = [gen2[2], gen2[3]]
    return grid, rand, first, tpe, parents, gen2, par2, gen3


def grid_random():
    grid, rand, *_ = trials()
    kinds = lambda pts: [
        ([p for p in pts if not inside(p)], "grey"),
        ([p for p in pts if inside(p)], "v"),
    ]
    a = panel(130, 60, 660, 380, "grid: 16 trials", kinds(grid), label=False)
    b = panel(970, 60, 660, 380, "random: 16 trials", kinds(rand), label=False)
    return slide(
        "grid-random",
        3,
        ["3.6"],
        'Grid and random: where do <span class="v">16 trials</span> go?',
        a
        + f'<ellipse cx="880" cy="230" rx="34" ry="26" fill="{PT}" fill-opacity="0.7"/>'
        + txt(880, 290, "good", 24, P, "middle", 600)
        + build(b),
        credit="After Bergstra &amp; Bengio (2012), Random search for hyper-parameter optimization.",
        mk="schematic",
        h=500,
    )


def tpe_slide():
    _, _, first, tpe, *_ = trials()
    x0, y0, w, h = 330, 50, 1060, 380
    a = panel(x0, y0, w, h, "", [(first, "hollow")])
    leg = circ(1440, 120, 12, BG, GREY, 4) + txt(1464, 130, "first trials", 26, MUT)
    b = (
        "".join(circ(x0 + w * u, y0 + h * (1 - v), 12, V, V, 4) for u, v in tpe)
        + circ(1440, 180, 12, V, V, 4)
        + txt(1464, 190, "later trials", 26, V, weight=600)
    )
    return slide(
        "tpe",
        3,
        ["3.6"],
        'TPE: the next trial goes where <span class="v">earlier trials did well</span>',
        a + leg + build(b),
        "TPE splits the trials so far into good and rest, and draws the next one where the good ones lie. Random search never looks back.",
        mk="schematic",
        h=490,
    )


def population_slide():
    _, _, first, _, parents, gen2, par2, gen3 = trials()
    x0, y0, w, h = 330, 50, 1060, 380
    XY = lambda p: (x0 + w * p[0], y0 + h * (1 - p[1]))
    a = panel(x0, y0, w, h, "", [(first, "hollow")])
    ring = lambda ps, c: "".join(circ(*XY(p), 22, "none", c, 3) for p in ps)
    link = lambda ps, kids, c: "".join(
        line(*XY(ps[min(i // 3, 1)]), *XY(k), c, 2, "5 5") for i, k in enumerate(kids)
    )
    b = (
        ring(parents, P)
        + link(parents, gen2, P)
        + "".join(circ(*XY(p), 12, P, P, 4) for p in gen2)
    )
    c = (
        ring(par2, V)
        + link(par2, gen3, V)
        + "".join(circ(*XY(p), 12, V, V, 4) for p in gen3)
    )
    leg = circ(1440, 120, 12, BG, GREY, 4) + txt(1464, 130, "generation 1", 26, MUT)
    return slide(
        "population",
        3,
        ["3.6"],
        'A population: keep the best, <span class="v">breed</span> the next from them',
        a
        + leg
        + build(
            b
            + circ(1440, 180, 12, P, P, 4)
            + txt(1464, 190, "generation 2", 26, P, weight=600)
        )
        + build(
            c
            + circ(1440, 240, 12, V, V, 4)
            + txt(1464, 250, "generation 3", 26, V, weight=600)
        ),
        "The ringed trials are kept; each new generation is made from small changes to them. Good candidates stay in the search instead of being drawn once.",
        mk="schematic",
        h=490,
    )


def space_grows():
    x0, y0, sc, budget = 330, 470, 0.58, 27
    s = line(x0 - 40, y0, 1500, y0, MUT, 2.5)
    for i in range(6):
        n = 3 ** (i + 1)
        x = x0 + i * 180
        covered = min(n, budget)
        g = rect(x, y0 - covered * sc, 110, covered * sc, P)
        if n > budget:
            g += rect(
                x + 1.5, y0 - n * sc + 1.5, 107, (n - budget) * sc - 1.5, VT, V, 3
            )
        g += txt(
            x + 55,
            y0 - max(n, budget) * sc - 16,
            f"{n:,}",
            32,
            V if n > budget else P,
            "middle",
            600,
        )
        g += txt(x + 55, y0 + 40, str(i + 1), 28, INK, "middle", 500)
        frac = "all tried" if n <= budget else f"1 in {n // budget}"
        g += txt(x + 55, y0 + 78, frac, 26, MUT, "middle")
        s += g if i < 3 else build(g)
    s += line(x0 - 40, y0 - budget * sc, 1500, y0 - budget * sc, INK, 3, "12 8") + txt(
        1516, y0 - budget * sc + 9, "budget: 27 trials", 28, INK, weight=500
    )
    s += txt(x0 - 60, 60, "combinations ↑", 28, MUT, "end")
    s += txt(x0 - 60, y0 + 40, "hyperparameters", 28, INK, "end", 500)
    return slide(
        "space-grows",
        3,
        ["3.7"],
        'Every additional hyperparameter <span class="v">multiplies</span> the search space',
        s,
        "With three values each, every new hyperparameter triples the number of combinations. The budget stays where it was.",
        h=570,
    )


def halving():
    y0, ep = 470, 44
    x0, pitch, cw = 150, 32, 24
    heights = [9] * 3 + [3] * 6 + [1] * 18
    s = ""
    for e in (1, 3, 9):
        s += line(
            x0 - 20, y0 - e * ep, x0 + 27 * pitch, y0 - e * ep, RULE, 2.5, "8 8"
        ) + txt(x0 - 30, y0 - e * ep + 9, f"epoch {e}", 26, MUT, "end")
    for i, hgt in enumerate(heights):
        s += rect(x0 + i * pitch, y0 - hgt * ep, cw, hgt * ep, V if hgt == 9 else GREY)
    s += line(x0 - 20, y0, x0 + 27 * pitch, y0, MUT, 2.5)
    s += txt(
        x0, y0 + 44, "27 trials start, 3 reach epoch 9", 30, INK, weight=500
    ) + txt(x0, y0 + 84, "18 + 18 + 27 = 63 epochs", 28, MUT)
    bx = 1260
    b = "".join(rect(bx + i * pitch, y0 - 9 * ep, cw, 9 * ep, P) for i in range(7))
    b += line(bx - 20, y0, bx + 7 * pitch + 10, y0, MUT, 2.5)
    b += txt(bx, y0 + 44, "7 trials, all to epoch 9", 30, INK, weight=500) + txt(
        bx, y0 + 84, "7 × 9 = 63 epochs", 28, MUT
    )
    b += txt(1150, 280, "=", 80, MUT, "middle", 500)
    return slide(
        "halving",
        3,
        ["3.7"],
        'Successive <span class="v">halving</span>',
        s + build(b),
        "Bad trials don't get the same epochs as good trials.",
        h=570,
    )


def slow_starter():
    x0, y0, w, h = 330, 480, 1050, 430
    X = lambda e: x0 + w * (e - 1) / 19
    Y = lambda v: y0 - h * v
    s = line(x0, y0, x0 + w + 40, y0, MUT, 2.5, marker="g") + line(
        x0, y0, x0, y0 - h - 30, MUT, 2.5, marker="g"
    )
    s += txt(x0 + w + 40, y0 + 86, "epoch →", 28, MUT, "end") + txt(
        x0 - 20, y0 - h - 10, "validation accuracy ↑", 28, MUT, "end"
    )
    for e in (1, 3, 9, 20):
        s += line(X(e), y0, X(e), y0 + 10, MUT, 2) + txt(
            X(e), y0 + 44, str(e), 26, MUT, "middle"
        )
    rise = lambda top, tau, base=0.0: (
        lambda e: base + (top - base) * (1 - math.exp(-e / tau))
    )
    rnd = random.Random(8)
    slow = rise(0.95, 6.5, 0.02)
    others = [rise(0.18 + 0.54 * i / 14, rnd.uniform(0.9, 1.6)) for i in range(15)]
    # successive halving: the lower half stops at epoch 3, half of the rest at epoch 9
    alive, end = [*others, slow], {}
    for rung in (3, 9):
        alive.sort(key=lambda f: f(rung))
        for f in alive[: len(alive) // 2]:
            end[f] = rung
        alive = alive[len(alive) // 2 :]
    cross = lambda x, y, c, r=7, wd=3: (
        line(x - r, y - r, x + r, y + r, c, wd)
        + line(x - r, y + r, x + r, y - r, c, wd)
    )
    for rung in (3, 9):
        s += line(X(rung), y0, X(rung), y0 - h, RULE, 2.5, "8 8")
    best = max(others, key=lambda f: f(20))
    for f in others:
        last = end.get(f, 20)
        col, wd = (P, 6) if f is best else (GREY, 2.5)
        s += poly([(X(e), Y(f(e))) for e in range(1, last + 1)], col, wd)
        if last < 20:
            s += cross(X(last), Y(f(last)), MUT)
    stop = end[slow]
    s += poly([(X(e), Y(slow(e))) for e in range(1, stop + 1)], GREY, 2.5)
    s += cross(X(stop), Y(slow(stop)), MUT)
    b = poly([(X(e), Y(slow(e))) for e in range(1, stop + 1)], V, 6)
    b += circ(X(stop), Y(slow(stop)), 22, BG, V, 4) + cross(
        X(stop), Y(slow(stop)), V, 10, 5
    )
    b += poly([(X(e), Y(slow(e))) for e in range(stop, 21)], V, 4, "10 8")
    return slide(
        "slow-starter",
        3,
        ["3.7"],
        'The pitfall: a <span class="v">slow starter</span> is stopped',
        s + build(b),
        "Halving stops the lower half at epoch 3 and half of the rest at epoch 9. A trial that starts slowly is stopped at epoch 3, however high it would have ended.",
        mk="schematic",
        h=570,
    )


def lr_valley():
    def valley(cx, us, col, title):
        Y = lambda u: 440 - 380 * u * u
        X = lambda u: cx + 330 * u
        g = txt(cx, 22, title, 30, INK, "middle", 500)
        g += poly([(X(-1 + i / 30), Y(-1 + i / 30)) for i in range(61)], MUT, 4)
        for a, b in itertools.pairwise(us):
            # a hop: an arc over the valley that ends just short of the next dot
            x1, y1, x2, y2 = X(a), Y(a), X(b), Y(b)
            span = abs(x2 - x1)
            qx, qy = (x1 + x2) / 2, min(y1, y2) - min(0.3 * span, 56) - 10
            d = math.hypot(x2 - qx, y2 - qy)
            ex, ey = x2 - 16 * (x2 - qx) / d, y2 - 16 * (y2 - qy) / d
            head = (
                f' marker-end="url(#m{"v" if col == V else "g"})"' if span > 40 else ""
            )
            g += (
                f'<path d="M {x1:.1f} {y1:.1f} Q {qx:.1f} {qy:.1f} {ex:.1f} {ey:.1f}" '
                f'fill="none" stroke="{col}" stroke-width="4"{head}/>'
            )
        g += "".join(circ(X(u), Y(u), 10, col) for u in us)
        return g

    a = valley(
        430,
        [0.86, -0.8, 0.92, -0.72, 0.8, -0.9, 0.74, -0.84],
        MUT,
        "the same learning rate every step",
    )
    b = valley(
        1290,
        [0.92, -0.62, 0.38, -0.2, 0.09, -0.03, 0.0],
        V,
        "a learning rate that shrinks",
    )
    return slide(
        "lr-valley",
        3,
        ["3.8"],
        'Gradually <span class="v">decrease</span> the learning rate',
        a + build(b),
        mk="schematic",
        h=480,
    )


def schedule_or_race():
    x0, y0, w, h = 140, 420, 640, 320
    X = lambda u: x0 + w * u
    Y = lambda v: y0 - h * v
    green = OK[2]
    s = txt(
        x0,
        34,
        "a scheduler: one training, the learning rate changes",
        30,
        INK,
        weight=500,
    )
    s += line(x0, y0, x0 + w + 30, y0, MUT, 2.5, marker="g") + line(
        x0, y0, x0, y0 - h - 20, MUT, 2.5, marker="g"
    )
    s += txt(x0 + w + 30, y0 + 44, "epochs →", 26, MUT, "end") + txt(
        x0 + 16, y0 - h - 24, "learning rate ↑", 26, MUT
    )
    s += poly([(X(0), Y(0.9)), (X(1), Y(0.9))], GREY, 5) + txt(
        X(1) + 14, Y(0.9) + 9, "constant", 26, MUT, weight=500
    )
    steps = [(0, 0.9), (0.33, 0.9), (0.33, 0.45), (0.66, 0.45), (0.66, 0.2), (1, 0.2)]
    s += poly([(X(u), Y(v)) for u, v in steps], P, 5)
    s += txt(X(1) + 14, Y(0.2) + 9, "step", 26, P, weight=600)
    s += poly(
        [(X(i / 40), Y(0.9 * math.exp(-4.5 * i / 40))) for i in range(41)], green, 5
    )
    s += txt(X(0.13), Y(0.33), "exponential", 26, green, "end", 600)
    cos = lambda u: 0.9 * (1 + math.cos(math.pi * u)) / 2
    s += poly([(X(i / 40), Y(cos(i / 40))) for i in range(41)], V, 6)
    s += txt(X(0.47), Y(0.66), "cosine", 26, V, weight=600)
    rx = 1040
    RX = lambda e: rx + 560 * (e - 1) / 19
    b = txt(rx, 34, "a race: whole trials stop", 30, INK, weight=500)
    b += line(rx, y0, rx + 590, y0, MUT, 2.5, marker="g") + line(
        rx, y0, rx, y0 - h - 20, MUT, 2.5, marker="g"
    )
    b += txt(rx + 590, y0 + 44, "epochs →", 26, MUT, "end") + txt(
        rx + 16, y0 - h - 24, "accuracy ↑", 26, MUT
    )
    for i in range(9):
        end = 20 if i == 0 else (12 if i < 3 else 6)
        top = 0.92 - 0.09 * i
        pts = [(RX(e), Y(top * (1 - math.exp(-e / 3)))) for e in range(1, end + 1)]
        col = V if end == 20 else GREY
        b += poly(pts, col, 5 if end == 20 else 3)
        if end < 20:
            x, y = pts[-1]
            b += line(x - 8, y - 8, x + 8, y + 8, INK, 3) + line(
                x - 8, y + 8, x + 8, y - 8, INK, 3
            )
    return slide(
        "schedule-or-race",
        3,
        ["3.8"],
        'A scheduler and a race are <span class="v">two different things</span>',
        s + build(b),
        "A scheduler lowers the learning rate inside one training. A race such as halving compares trials and stops whole ones.",
        mk="schematic",
        h=480,
    )


# ---------------------------------------------------------------- part 4
def pareto_front():
    sec = taken("lesson3-tradeoffs.html", "pareto-front", 4, ["3.9", "3.10"])
    first = sec.index("</g>", sec.index("<g data-build-in><rect")) + 4
    inner = build(
        rect(220, 250, 800, 230, PT, P, 2, dash="8 6", op=0.45)
        + txt(
            220,
            530,
            "inside: can be faster at the same score, or score higher at the same speed",
            28,
            P,
            weight=500,
        )
    )
    return sec[:first] + inner + sec[first:]


# ---------------------------------------------------------------- your study
MIX = [
    ("dishes", "Food-101", 75750, 101),
    ("traffic signs", "GTSRB", 26640, 43),
    ("satellite tiles", "EuroSAT", 21600, 10),
    ("galaxies", "Galaxy10", 15962, 10),
    ("blood cells", "BloodMNIST", 13671, 8),
    ("flowers", "Oxford Flowers", 7169, 102),
    ("textures", "DTD", 3760, 47),
]


def mix7():
    share = 0.25
    s = txt(480, 30, "training images", 26, MUT)
    s += txt(1450, 30, "classes", 26, MUT, "end") + txt(
        1700, 30, "images per class", 26, MUT, "end"
    )
    cut = ""
    for i, (kind, name, n, k) in enumerate(MIX):
        y = 50 + i * 62
        wd = 720 * n / 75750
        s += txt(450, y + 30, kind, 32, INK, "end", 500) + txt(
            200, y + 30, name, 24, MUT, "end"
        )
        s += rect(480, y, wd, 42, PT) + txt(
            480 + wd + 16, y + 31, f"{n:,}", 28, P, weight=600
        )
        s += txt(1450, y + 31, str(k), 30, MUT, "end", 500)
        s += txt(1700, y + 31, f"{round(n / k):,}", 30, V, "end", 600)
        cut += rect(480, y, wd * share, 42, P)
    cut += txt(
        480,
        50 + 7 * 62 + 24,
        "share = 0.25: the same quarter of every class",
        28,
        P,
        weight=600,
    )
    return (
        '<section id="mix7" class="paper std" data-crumb="your study">'
        + slide(
            "mix7",
            5,
            ["3.12"],
            'The data: seven kinds of image, <span class="v">70 to 2,160</span> per class',
            s + build(cut),
            "A smaller <code>share</code> takes the same fraction of every class, so a smaller run still holds all seven kinds and all 321 classes.",
            h=520,
        ).split(">", 1)[1]
    )


def your_front():
    x0, y0, w, h = 300, 470, 1060, 420
    X = lambda u: x0 + w * u
    Y = lambda v: y0 - h * v
    lo, hi = 0.3, 0.62
    pts = [
        (0.06, 0.2), (0.12, 0.34), (0.2, 0.3), (0.24, 0.5), (0.34, 0.42), (0.36, 0.63),
        (0.43, 0.55), (0.47, 0.72), (0.52, 0.6), (0.58, 0.78), (0.6, 0.66), (0.7, 0.7),
        (0.76, 0.83), (0.84, 0.76), (0.92, 0.86),
    ]  # fmt: skip
    band = [p for p in pts if lo <= p[0] <= hi]
    front = [
        p
        for p in band
        if not any(q[0] <= p[0] and q[1] >= p[1] and q != p for q in band)
    ]
    s = line(x0, y0, x0 + w + 40, y0, MUT, 2.5, marker="g") + line(
        x0, y0, x0, y0 - h - 30, MUT, 2.5, marker="g"
    )
    s += txt(
        x0 + w + 40, y0 + 50, "cost: FLOPs per image (log) →", 28, MUT, "end"
    ) + txt(x0 - 20, y0 - h - 10, "accuracy ↑", 28, MUT, "end")
    s += "".join(circ(X(u), Y(v), 13, BG, GREY, 4) for u, v in pts)
    b = rect(X(lo), y0 - h, X(hi) - X(lo), h, PT, op=0.55)
    b += txt((X(lo) + X(hi)) / 2, y0 - h - 14, "your band", 28, P, "middle", 600)
    b += "".join(circ(X(u), Y(v), 13, PT, P, 4) for u, v in band)
    c = poly([(X(u), Y(v)) for u, v in sorted(front)], V, 5)
    c += "".join(circ(X(u), Y(v), 15, V, V, 3) for u, v in front)
    c += txt(X(hi) + 20, Y(0.9), "your front", 30, V, weight=600)
    return slide(
        "your-front",
        5,
        ["3.9", "3.11", "3.12"],
        'Your study: find <span class="v">your own Pareto front</span>',
        s + build(b) + build(c),
        "Test the scaling law on your own designs: within a band of compute you choose, find your own Pareto front, not the highest accuracy or the fastest model. Choose a route through the architectures you know and say why; a small model that wins on speed is a result too. A mixture of experts is allowed, but these vectors are close to linearly separable, which limits what it adds.",
        mk="schematic",
        h=540,
    )


def deck():
    out = cover() + agenda()
    out += divider(
        "p01",
        1,
        "01 scale",
        'More parameters, more data, more compute: what does it <span class="v">buy</span>?',
        [
            (
                'A <span class="v">scaling law</span>',
                "The loss falls as a power of what you spend.",
            ),
            (
                'Grow <span class="v">together</span>',
                "Parameters and data, not one of the two.",
            ),
            (
                'The <span class="v">control</span>',
                "Compare architectures at equal size.",
            ),
        ],
    )
    out += claim_or_size() + law_curves() + law_price() + grow_together()
    out += taken(
        "lesson3-scaling.html",
        "data-for-parameters",
        1,
        ["3.3"],
        title='To avoid overfitting, increase data <span class="v">~5×</span>',
    )
    out += gopher() + compute_area()
    out += taken(
        "lesson3-scaling.html",
        "chinchilla",
        1,
        ["3.3"],
        title='Gopher (280B) vs <span class="v">Chinchilla (70B)</span>',
    )
    out += bet_or_scale()
    out += divider(
        "p02",
        2,
        "02 experts",
        'A gate <span class="v">chooses</span> between routes',
        [
            ('The <span class="v">highway</span>', "A block, or its own input."),
            (
                'A <span class="v">mixture</span> of experts',
                "More routes, and a router.",
            ),
            (
                'Stored and <span class="v">used</span>',
                "Many parameters stored, few used per image.",
            ),
        ],
    )
    out += family() + highway_g() + highway_to_moe()
    out += taken("lesson3-moe.html", "moe-parameters", 2, ["3.5"])
    out += moe_split() + moe_collapse()
    out += divider(
        "p03",
        3,
        "03 hyperparameters",
        'Hyperparameter optimization: an algorithm chooses the <span class="v">next trial</span>',
        [
            (
                'The <span class="v">next trial</span>',
                "Grid, random, TPE, a population.",
            ),
            ('A <span class="v">race</span>', "Many trials short, the best ones long."),
            (
                'Inside <span class="v">one training</span>',
                "A learning rate that goes down.",
            ),
        ],
    )
    out += (
        grid_random()
        + tpe_slide()
        + population_slide()
        + space_grows()
        + halving()
        + slow_starter()
        + lr_valley()
        + schedule_or_race()
    )
    out += divider(
        "p04",
        4,
        "04 pareto",
        'Pareto fronts: which model is <span class="v">best</span>, if you care about two things?',
        [
            ('<span class="v">Beaten</span>', "Another model is better on both."),
            ('The <span class="v">front</span>', "The models nothing beats."),
            (
                'Your <span class="v">choice</span>',
                "Which point depends on what you need.",
            ),
        ],
    )
    out += taken("lesson3-tradeoffs.html", "pareto-beaten", 4, ["3.9"]) + pareto_front()
    out += mix7() + your_front()
    return out


def main():
    src = (DECKS / "les2.html").read_text()
    head = (
        src[: src.index("<section")]
        .replace("Lesson 2 · Kernels, state and gates", "Lesson 3 · Scaling and search")
        .replace('data-lesson="2"', 'data-lesson="3"')
    )
    head = head.replace(
        '<div id="stage">',
        f'<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>{MARKERS}</defs></svg>\n<div id="stage">',
        1,
    )
    tail = src[src.rindex("</section>") + len("</section>") :]
    cut = tail.index("<script>(function(){\n  // instruments")
    tail = tail[:cut] + "</body>\n</html>\n"
    body = deck()
    (DECKS / "les3.html").write_text(head + body + tail)
    print(body.count("<section"), "slides")


main()
