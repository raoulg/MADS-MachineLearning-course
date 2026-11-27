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
    3: ("03 — search on a budget", "notebook 3.3"),
    4: ("04 — trade-offs", "exercise 3.5"),
    5: ("05 — your study", "exercise 3.5"),
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
    s = f'\n    <p class="s" data-build-in>{sentence}</p>' if sentence else ""
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
    return sec + "\n\n"


# ---------------------------------------------------------------- cover and map
def glyph(kind):
    if kind == 1:
        g = poly([(10, 20), (70, 60), (150, 95), (250, 110)], "#86b9c9", 5) + "".join(
            circ(x, y, 7, "#86b9c9")
            for x, y in ((10, 20), (70, 60), (150, 95), (250, 110))
        )
    elif kind == 2:
        g = (
            ribbon(0, 65, 90, 65, 50, "#86b9c9")
            + ribbon(90, 47, 250, 20, 14, "#f2643c")
            + ribbon(90, 65, 250, 65, 22, "#f2643c")
            + ribbon(90, 83, 250, 110, 14, "#f2643c")
        )
    elif kind == 3:
        rnd = random.Random(4)
        g = (
            rect(100, 0, 40, 130, "#86b9c9", op=0.25)
            + "".join(
                circ(rnd.uniform(5, 255), rnd.uniform(8, 122), 7, "#b8b0a5")
                for _ in range(9)
            )
            + "".join(
                circ(x, y, 7, "#f2643c") for x, y in ((112, 30), (126, 70), (118, 104))
            )
        )
    elif kind == 4:
        fr = [(30, 15), (110, 30), (180, 60), (230, 105)]
        g = (
            "".join(
                circ(x, y, 7, "#b8b0a5")
                for x, y in ((40, 80), (90, 100), (120, 75), (160, 110), (70, 55))
            )
            + poly(fr, "#f2643c", 5)
            + "".join(circ(x, y, 8, "#f2643c") for x, y in fr)
        )
    else:
        g = (
            "".join(rect(x, 45, 60, 40, "none", "#86b9c9", 3) for x in (0, 95, 190))
            + line(62, 65, 91, 65, "#b8b0a5", 3)
            + line(157, 65, 186, 65, "#b8b0a5", 3)
            + poly([(220, 88), (220, 118), (30, 118), (30, 90)], "#f2643c", 3)
        )
    return f'<svg viewBox="-5 -5 270 140" style="width:270px; height:140px" aria-hidden="true">{g}</svg>'


def cover():
    art = (
        poly([(0, 40), (150, 150), (330, 235), (600, 290)], "#86b9c9", 6)
        + "".join(
            circ(x, y, 12, "#141210", "#86b9c9", 5)
            for x, y in ((0, 40), (150, 150), (330, 235))
        )
        + circ(600, 290, 14, "#f2643c")
    )
    return f"""<section id="cover" class="ink" style="display:flex; flex-direction:column; padding:236px 100px 56px">
  <img src="../img/lesson1/band_title_ink.png" alt="pttrn logo in a Rule 30 band" style="position:absolute; left:0; top:0; width:1920px; height:201px; object-fit:contain">
  <svg viewBox="-20 0 660 340" style="position:absolute; top:300px; right:120px; width:660px; height:340px" aria-hidden="true">{art}</svg>
  <div class="kick"><p style="color:#b8b0a5">MADS-ML</p><p style="color:#b8b0a5">Machine Learning</p></div>
  <div style="flex:1"></div>
  <div style="display:flex; flex-direction:column; gap:32px">
    <div style="width:280px; height:2px; background:#f2643c"></div>
    <h1 style="font-family:Outfit, Arial, sans-serif; font-size:132px; font-weight:500; line-height:0.98; letter-spacing:-4px; color:#f4f1ec">Scaling<br>and search</h1>
    <p style="max-width:1000px; font-family:Outfit, Arial, sans-serif; font-size:40px; line-height:1.3; color:#b8b0a5">What did that capacity buy you?</p>
  </div>
  <div class="foot" style="border-top-color:#2b2621"><p style="color:#b8b0a5">Lesson 3</p><p style="color:#b8b0a5">Hogeschool Utrecht</p></div>
</section>

"""


def agenda():
    parts = [
        (1, "Scale", "what more parameters and data buy"),
        (2, "Mixture of experts", "many parameters stored, few used"),
        (3, "Search on a budget", "where the trials go, and how long"),
        (4, "Trade-offs", "which model is best"),
        (5, "Your study", "what an image costs and buys"),
    ]
    cells = "".join(
        f'<div class="top{" first" if n == 1 else ""}" style="gap:18px"><p class="ck2">0{n}</p>'
        f'<div style="background:#141210; padding:14px 20px">{glyph(n)}</div>'
        f'<p class="s" style="font-size:34px; font-weight:500">{name}</p><p class="m" style="font-size:26px">{q}</p></div>'
        for n, name, q in parts
    )
    return f"""<section id="map" class="paper std" data-crumb="agenda">
  <img class="strip" src="../img/lesson1/band_strip_paper.png" alt="pttrn logo in a Rule 30 strip">
  <div class="kick"><p>Agenda</p><p>five parts</p><p><a href="../leerdoelen/les3.html">all goals</a></p></div>
  <div class="body" style="gap:60px">
    <h2 class="t">Today: what did that <span class="v">capacity</span> buy you?</h2>
    <div style="display:grid; grid-template-columns:repeat(5,1fr); gap:36px">{cells}</div>
  </div>
  <p class="pn n"></p>
</section>

"""


# ---------------------------------------------------------------- part 1
def claim_or_size():
    x0, full, bh = 430, 1000, 76
    w = lambda k: full * k / 120
    row = lambda y, name: txt(x0 - 30, y + 50, name, 34, INK, "end", 500)
    s = (
        row(50, "with gate")
        + rect(x0, 50, w(120), bh, P)
        + txt(x0 + 20, 100, "120k parameters", 30, "#ffffff", weight=500)
    )
    s += (
        row(200, "baseline")
        + rect(x0, 200, w(75), bh, P)
        + txt(x0 + 20, 250, "75k parameters", 30, "#ffffff", weight=500)
    )
    s += rect(x0 + w(75), 200, w(45), bh, VT, V, 3, dash="12 8") + txt(
        x0 + w(97.5), 250, "45k fewer", 30, V, "middle", 600
    )
    s += build(
        row(350, "control")
        + rect(x0, 350, w(120), bh, P)
        + txt(x0 + 20, 400, "120k parameters, no gate", 30, "#ffffff", weight=500)
        + line(x0 + w(120), 40, x0 + w(120), 440, V, 3, "10 8")
        + txt(x0 + w(120) + 24, 250, "equal size:", 30, V, weight=600)
        + txt(x0 + w(120) + 24, 290, "only the gate", 30, V, weight=600)
        + txt(x0 + w(120) + 24, 330, "differs", 30, V, weight=600)
    )
    return slide(
        "claim-or-size",
        1,
        ["3.1"],
        'Is it the gate, or the <span class="v">extra parameters</span>?',
        s,
        "A baseline with fewer parameters may lose only because it is smaller. The control is the same model without the new part, with as many parameters.",
        h=470,
    )


def law_loglog():
    a = {"parameters": 0.076, "data": 0.095, "compute": 0.050}
    # left panel: linear axes
    lx0, ly0, lw, lh = 110, 470, 640, 400
    s = line(lx0, ly0, lx0 + lw + 40, ly0, MUT, 2.5, marker="g") + line(
        lx0, ly0, lx0, ly0 - lh - 30, MUT, 2.5, marker="g"
    )
    s += txt(lx0 + lw / 2, 30, "linear axes", 30, INK, "middle", 500)
    s += txt(lx0 + lw + 40, ly0 + 46, "parameters →", 28, MUT, "end") + txt(
        lx0 + 16, 62, "test loss ↑", 28, MUT
    )
    f = lambda n: n ** -a["parameters"]
    yl = lambda v: ly0 - lh * (v - 0.5) / 0.55
    pts = [
        (lx0 + lw * (n - 1) / 999, yl(f(n)))
        for n in [1 + 999 * (i / 200) ** 3 for i in range(201)]
    ]
    s += poly(pts, P, 6)
    for n, lab in ((1, "×1"), (500, "×500"), (1000, "×1000")):
        x = lx0 + lw * (n - 1) / 999
        s += line(x, ly0, x, ly0 + 10, MUT, 2) + txt(
            x,
            ly0 + 46,
            lab if n != 1000 else "",
            26,
            MUT,
            "middle",
        )
    # right panel: log-log
    rx0, rw = 930, 520
    xl = lambda n: rx0 + rw * math.log10(n) / 3
    ylog = lambda v: (
        ly0
        - lh * (math.log10(v) - math.log10(0.5)) / (math.log10(1.05) - math.log10(0.5))
    )
    r = line(rx0, ly0, rx0 + rw + 40, ly0, MUT, 2.5, marker="g") + line(
        rx0, ly0, rx0, ly0 - lh - 30, MUT, 2.5, marker="g"
    )
    r += txt(rx0 + rw / 2, 30, "log-log axes", 30, INK, "middle", 500)
    for n in (1, 10, 100, 1000):
        r += line(xl(n), ly0, xl(n), ly0 + 10, MUT, 2) + txt(
            xl(n), ly0 + 46, f"×{n}", 26, MUT, "middle"
        )
    r += poly([(xl(1), ylog(1)), (xl(1000), ylog(f(1000)))], P, 6)
    for n in (1, 10, 100, 1000):
        r += circ(xl(n), ylog(f(n)), 11, BG, P, 5)
    for n in (1, 10, 100):
        r += poly(
            [
                (xl(n), ylog(f(n))),
                (xl(n * 10), ylog(f(n))),
                (xl(n * 10), ylog(f(n * 10))),
            ],
            V,
            3,
            "8 6",
        )
        if n == 1:
            r += txt(
                xl(10) + 14, ylog(f(1)) - 4, "× 0.84 for every 10×", 26, V, weight=600
            )
    r += txt(xl(1000) + 22, ylog(f(1000)) + 10, "parameters", 28, P, weight=600)
    more = ""
    for k, col, fac in (("compute", GREY, "× 0.89"), ("data", GREY, "× 0.80")):
        g = lambda n, k=k: n ** -a[k]
        more += poly([(xl(1), ylog(1)), (xl(1000), ylog(g(1000)))], col, 4)
        more += txt(
            xl(1000) + 22, ylog(g(1000)) + 10, f"{k} {fac}", 28, MUT, weight=500
        )
    return slide(
        "law-loglog",
        1,
        ["3.2"],
        'A scaling law: the loss against <span class="v">10× more</span>',
        s + build(r) + build(more),
        "Kaplan et al. (2020) trained language models of many sizes: every 10× more parameters multiplies the test loss by 0.84. On log-log axes that is a straight line.",
        "Kaplan et al. (2020), Scaling laws for neural language models.",
        h=530,
    )


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


def law_range():
    rows = sorted(
        (
            r
            for r in json.loads((RESULTS / "food_scaling.json").read_text())
            if r["width"] == 0
        ),
        key=lambda r: r["n"],
    )
    ns, ls = [r["n"] for r in rows], [r["held"] for r in rows]
    lx, ly = [math.log10(n) for n in ns[:3]], [math.log10(v) for v in ls[:3]]
    mx, my = sum(lx) / 3, sum(ly) / 3
    slope = sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum(
        (a - mx) ** 2 for a in lx
    )
    fit = lambda n: 10 ** (my + slope * (math.log10(n) - mx))
    x0, y0, w, h = 330, 480, 1000, 430
    X = lambda n: (
        x0
        + w * (math.log10(n) - math.log10(700)) / (math.log10(110000) - math.log10(700))
    )
    Y = lambda v: (
        y0
        - h * (math.log10(v) - math.log10(0.15)) / (math.log10(0.6) - math.log10(0.15))
    )
    s = rect(X(ns[0]) - 40, Y(0.6), X(ns[2]) - X(ns[0]) + 80, y0 - Y(0.6), PT, op=0.45)
    s += txt(X(ns[1]), Y(0.6) + 36, "the line is fitted here", 26, P, "middle", 500)
    s += line(x0, y0, x0 + w + 40, y0, MUT, 2.5, marker="g") + line(
        x0, y0, x0, y0 - h - 30, MUT, 2.5, marker="g"
    )
    s += txt(x0 + w + 40, y0 + 86, "training images (log) →", 28, MUT, "end") + txt(
        x0 - 20, y0 - h - 10, "test loss (log) ↑", 28, MUT, "end"
    )
    for n, lab in ((1000, "1,000"), (10000, "10,000"), (100000, "100,000")):
        s += line(X(n), y0, X(n), y0 + 10, MUT, 2) + txt(
            X(n), y0 + 44, lab, 26, MUT, "middle"
        )
    for v in (0.2, 0.3, 0.5):
        s += line(x0 - 10, Y(v), x0, Y(v), MUT, 2) + txt(
            x0 - 20, Y(v) + 9, f"{v}", 26, MUT, "end"
        )
    s += poly([(X(ns[0]), Y(fit(ns[0]))), (X(ns[2]), Y(fit(ns[2])))], P, 5)
    s += "".join(circ(X(n), Y(v), 13, BG, P, 5) for n, v in zip(ns[:3], ls[:3]))
    b = poly([(X(ns[2]), Y(fit(ns[2]))), (X(ns[4]), Y(fit(ns[4])))], P, 4, "12 9")
    b += line(X(ns[4]), Y(fit(ns[4])), X(ns[4]), Y(ls[4]) + 16, V, 5)
    b += "".join(circ(X(n), Y(v), 13, VT, V, 5) for n, v in zip(ns[3:], ls[3:]))
    b += txt(X(ns[4]) + 28, Y(ls[4]) + 10, f"measured {ls[4]:.2f}", 30, V, weight=600)
    b += txt(
        X(ns[4]) + 28,
        Y(fit(ns[4])) + 10,
        f"line promises {fit(ns[4]):.2f}",
        30,
        P,
        weight=600,
    )
    return slide(
        "law-range",
        1,
        ["3.2", "3.10"],
        'Our own curve: does the line hold <span class="v">further on</span>?',
        s + build(b),
        f"A line through the three smallest data sizes promises {fit(ns[4]):.2f} at 70,700 images; the run gives {ls[4]:.2f}. A scaling law holds inside the range it was measured on.",
        "Food-101, a linear head on a frozen encoder · notebook 3.1.3",
        h=570,
    )


def grow_together():
    cw, ch = 600, 220
    col = {0: 360, 1: 1060}
    rowy = {1: 20, 0: 300}  # 1 = big model on top

    def cell(c, r, train, valid, label, tint, lab_col, gap=False):
        x, y = col[c], rowy[r]
        X = lambda u: x + 30 + (cw - 60) * u
        Y = lambda v: y + ch - 20 - (ch - 40) * v
        us = [i / 40 for i in range(41)]
        g = rect(x, y, cw, ch, tint, RULE, 2)
        if gap:
            area = [(X(u), Y(train(u))) for u in us] + [
                (X(u), Y(valid(u))) for u in reversed(us)
            ]
            g += poly(area, "none", 0, fill=VT)
        g += poly([(X(u), Y(train(u))) for u in us], GREY, 4)
        g += poly([(X(u), Y(valid(u))) for u in us], P, 6)
        g += txt(x + cw - 24, y + 44, label, 32, lab_col, "end", 600)
        return g

    e = math.exp
    s = txt(col[0] + cw / 2, 560, "little data", 32, INK, "middle", 500) + txt(
        col[1] + cw / 2, 560, "much data", 32, INK, "middle", 500
    )
    s += txt(330, rowy[1] + ch / 2 + 10, "big model", 32, INK, "end", 500) + txt(
        330, rowy[0] + ch / 2 + 10, "small model", 32, INK, "end", 500
    )
    s += (
        line(40, 596, 70, 596, GREY, 4)
        + txt(80, 605, "train loss", 26, MUT)
        + line(240, 596, 270, 596, P, 6)
        + txt(280, 605, "validation loss", 26, MUT)
    )
    s += cell(
        0,
        0,
        lambda u: 0.42 + 0.5 * e(-4 * u),
        lambda u: 0.5 + 0.42 * e(-4 * u),
        "learns",
        BG,
        MUT,
    )
    s += build(
        cell(
            0,
            1,
            lambda u: 0.04 + 0.88 * e(-5 * u),
            lambda u: 0.42 + 0.5 * e(-9 * u) + 0.42 * u * u,
            "overfits",
            "#f9e6de",
            V,
            True,
        )
    )
    s += build(
        cell(
            1,
            0,
            lambda u: 0.4 + 0.52 * e(-10 * u),
            lambda u: 0.43 + 0.49 * e(-10 * u),
            "stuck at its floor",
            GREYT,
            MUT,
        )
    )
    s += build(
        cell(
            1,
            1,
            lambda u: 0.1 + 0.82 * e(-4 * u),
            lambda u: 0.14 + 0.78 * e(-4 * u),
            "lowest loss",
            "#e3eef2",
            P,
        )
        + line(
            col[0] + cw + 14,
            rowy[0] - 6,
            col[1] - 18,
            rowy[1] + ch + 10,
            V,
            8,
            marker="v",
        )
    )
    return slide(
        "grow-together",
        1,
        ["3.3"],
        'Parameters and data <span class="v">grow together</span>',
        s,
        "A big model on little data overfits; a small model on much data runs into its own floor. Kaplan (2020): 8× the parameters ask for about 5× the data.",
        mk="schematic",
        h=620,
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
    b += txt(
        x0 + 70 * sx + 24, y0 - 1400 * sy + 88, "about the same area", 30, V, weight=500
    )
    return slide(
        "compute-area",
        1,
        ["3.3"],
        'Compute is an area: <span class="v">parameters × data</span>',
        s + build(b),
        "Every token passes every parameter, so the compute of a training grows with parameters × data. One budget can be wide and flat, or narrow and tall.",
        "Rae et al. (2021), Gopher · Hoffmann et al. (2022), Chinchilla.",
        h=590,
    )


def bet_or_scale():
    x0, y0 = 300, 500
    s = line(x0, y0, 1560, y0, MUT, 2.5, marker="g") + line(
        x0, y0, x0, 30, MUT, 2.5, marker="g"
    )
    s += txt(
        1560, y0 + 50, "parameters, with the data they need (log) →", 28, MUT, "end"
    ) + txt(x0 - 20, 50, "loss ↑", 28, MUT, "end")
    ya = lambda x: 110 + (x - 360) * 0.175
    yb = lambda x: ya(x) + 105
    s += poly([(360, ya(360)), (1440, ya(1440))], P, 6) + txt(
        1460, ya(1440) + 10, "architecture A", 30, P, weight=600
    )
    s += circ(620, ya(620), 13, BG, P, 5) + circ(900, ya(900), 13, BG, P, 5)
    s += line(650, ya(650) - 30, 868, ya(868) - 30, INK, 4, marker="ink") + txt(
        640, ya(650) - 62, "scale: along the curve", 30, INK, "start", 500
    )
    b = poly([(360, yb(360)), (1440, yb(1440))], V, 6) + txt(
        1460, yb(1440) + 10, "architecture B", 30, V, weight=600
    )
    b += line(900, 40, 900, y0, MUT, 2.5, "10 8") + txt(
        912, 62, "compare here: equal parameters and data", 26, MUT
    )
    b += line(900, ya(900) + 18, 900, yb(900) - 20, V, 5, marker="v") + circ(
        900, yb(900), 13, VT, V, 5
    )
    b += txt(880, yb(900) + 78, "a better bet: a lower curve", 30, V, "end", 600)
    return slide(
        "bet-or-scale",
        1,
        ["3.1", "3.11"],
        'Scale, or a <span class="v">better bet</span>?',
        s + build(b),
        "More parameters and data move a model along its curve. An architecture whose bet fits the data lies on a lower curve, so compare two architectures at equal parameters and data.",
        mk="schematic",
        h=560,
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
    return sec + "\n\n"


def flow(x0, weights, colours, top, title, labels, box, cy=300, W=150, spread=None):
    """An input band that splits over routes of width weight × W and joins again."""
    k = len(weights)
    spread = spread or (130 if k > 2 else 150)
    ys = [cy + (i - (k - 1) / 2) * spread for i in range(k)]
    g = txt(x0, top, title, 30, INK, weight=500)
    g += rect(x0, cy - W / 2, 90, W, P) + txt(
        x0 + 45, cy + W / 2 + 40, "x", 30, INK, "middle", 500
    )
    off = cy - W / 2
    for i, (wt, c) in enumerate(zip(weights, colours)):
        w = wt * W
        a = off + w / 2
        off += w
        if w < 1:
            g += poly([(x0 + 90, cy), (x0 + 230, ys[i])], GREY, 2, "6 6") + line(
                x0 + 230, ys[i], x0 + 500, ys[i], GREY, 2, "6 6"
            )
            g += poly([(x0 + 500, ys[i]), (x0 + 640, cy)], GREY, 2, "6 6")
        else:
            g += ribbon(x0 + 90, a, x0 + 230, ys[i], w, c) + rect(
                x0 + 229, ys[i] - w / 2, 272, w, c
            )
            g += ribbon(x0 + 500, ys[i], x0 + 640, a, w, c) + rect(
                x0 + 639, a - w / 2, 92, w, c
            )
        name, wl = box[i], labels[i]
        if name:
            live = w >= 1
            g += rect(x0 + 290, ys[i] - 36, 130, 72, BG, V if live else GREY, 3) + txt(
                x0 + 355, ys[i] + 10, name, 26, V if live else GREY, "middle", 500
            )
        if k == 2:
            ly = ys[i] - max(w / 2, 36) - 16 if i == 0 else ys[i] + w / 2 + 36
            g += txt(x0 + 365, ly, wl, 26, INK, "middle", 500, mono=True)
        else:
            g += txt(
                x0 + 468,
                ys[i] - max(w / 2, 6) - 10,
                wl,
                26,
                INK if w >= 1 else GREY,
                "middle",
                500,
                mono=True,
            )
    g += txt(x0 + 685, cy + W / 2 + 40, "y", 30, INK, "middle", 500)
    return g


def highway_g():
    a = flow(
        60,
        [0.8, 0.2],
        [V, P],
        40,
        "one input: the gate is mostly open",
        ["g = 0.8", "skip: 1 − g = 0.2"],
        ["block", ""],
    )
    b = flow(
        930,
        [0.2, 0.8],
        [V, P],
        40,
        "another input: mostly closed",
        ["g = 0.2", "skip: 1 − g = 0.8"],
        ["block", ""],
    )
    shapes = txt(60 + 45, 300 - 95, "(B, D)", 24, P, "middle", mono=True) + txt(
        60 + 685, 300 - 95, "(B, D)", 24, P, "middle", mono=True
    )
    return slide(
        "highway-g",
        2,
        ["3.4"],
        'A highway: a gate <span class="v">divides the flow</span> over two routes',
        a + shapes + build(b),
        "The gate g weighs the block, 1 − g weighs the skip around it, and the two are added. Both routes must give the same shape, (B, D).",
        h=540,
    )


def highway_to_moe():
    names = ["expert 1", "expert 2", "expert 3", "expert 4"]
    a = flow(
        60,
        [0.15, 0.45, 0.3, 0.1],
        [V] * 4,
        30,
        "every expert, weighed",
        ["0.15", "0.45", "0.30", "0.10"],
        names,
        cy=310,
        spread=128,
    )
    b = flow(
        930,
        [0, 1, 0, 0],
        [V] * 4,
        30,
        "sparse: only the top expert runs",
        ["0", "1", "0", "0"],
        names,
        cy=310,
        spread=128,
    )
    router = lambda x: (
        rect(x - 10, 90, 150, 60, PT, P, 3)
        + txt(x + 65, 129, "router", 26, P, "middle", 500)
        + line(x + 45, 152, x + 45, 230, P, 3, "6 6")
    )
    return slide(
        "highway-to-moe",
        2,
        ["3.4", "3.5"],
        'A mixture of experts: a highway with <span class="v">more routes</span>',
        a + router(60) + build(b + router(930)),
        "A router gives every expert a weight for this input, and the weights sum to 1. A sparse mixture runs only the expert with the highest weight.",
        h=560,
    )


def moe_control():
    dense = [
        (386 * w + 321, acc)
        for w, acc in (
            (4, 0.281),
            (5, 0.360),
            (8, 0.497),
            (9, 0.525),
            (16, 0.649),
            (17, 0.660),
            (24, 0.702),
        )
    ]
    moe = [
        (2385, 0.514, 0.360, 2251, "+15"),
        (3929, 0.656, 0.525, 3795, "+13"),
        (7017, 0.726, 0.660, 6883, "+7"),
    ]
    x0, y0, w, h = 330, 480, 1000, 430
    X = lambda n: (
        x0
        + w
        * (math.log10(n) - math.log10(1600))
        / (math.log10(11000) - math.log10(1600))
    )
    Y = lambda v: y0 - h * (v - 0.2) / 0.6
    s = line(x0, y0, x0 + w + 40, y0, MUT, 2.5, marker="g") + line(
        x0, y0, x0, y0 - h - 30, MUT, 2.5, marker="g"
    )
    s += txt(
        x0 + w + 40, y0 + 86, "parameters used per image (log) →", 28, MUT, "end"
    ) + txt(x0 - 20, y0 - h - 10, "test accuracy ↑", 28, MUT, "end")
    for n, lab in ((2000, "2,000"), (4000, "4,000"), (8000, "8,000")):
        s += line(X(n), y0, X(n), y0 + 10, MUT, 2) + txt(
            X(n), y0 + 44, lab, 26, MUT, "middle"
        )
    for v in (0.3, 0.5, 0.7):
        s += line(x0 - 10, Y(v), x0, Y(v), MUT, 2) + txt(
            x0 - 20, Y(v) + 9, f"{v}", 26, MUT, "end"
        )
    s += poly([(X(n), Y(a)) for n, a in dense], P, 5) + "".join(
        circ(X(n), Y(a), 10, BG, P, 4) for n, a in dense
    )
    s += txt(X(9585) + 24, Y(0.702) + 10, "dense head", 30, P, weight=600)
    b = ""
    for n, acc, cacc, cn, lab in moe:
        b += line(X(n), Y(cacc) - 14, X(n), Y(acc) + 22, V, 4, marker="v") + circ(
            X(n), Y(acc), 15, VT, V, 5
        )
        b += txt(X(n) - 24, Y(acc) + 4, lab, 30, V, "end", 600)
    b += txt(X(7017) + 28, Y(0.726) - 24, "mixture of experts", 30, V, weight=600)
    return slide(
        "moe-control",
        2,
        ["3.1", "3.5"],
        'A mixture of experts against <span class="v">its control</span>',
        s + build(b),
        "At three budgets the mixture scores higher than the dense head that uses as many parameters per image. The gain shrinks as the budget grows.",
        "Seven kinds of image, 321 classes · notebook 3.2.4 · one seed",
        h=570,
    )


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


def moe_split():
    cw, ch = 72, 56

    def grid(x0, which, title):
        g = txt(x0, 34, title, 30, INK, weight=500)
        for r, (dom, shares) in enumerate(ROUTING[which].items()):
            y = 60 + r * ch
            g += txt(x0 - 20, y + 38, dom, 26, MUT, "end")
            for c, v in enumerate(shares):
                g += rect(x0 + c * cw, y, cw - 4, ch - 4, V, op=max(v, 0.02) ** 0.8)
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

    return slide(
        "moe-split",
        2,
        ["3.5"],
        'Where does the router <span class="v">send</span> each kind of image?',
        grid(230, "untrained", "before training")
        + build(grid(1040, "trained", "after training")),
        "Nobody tells the router which kind of image it reads. After training, galaxies, flowers and dishes each have an expert of their own.",
        "Share of each domain's test images per expert · notebook 3.2.5",
        h=510,
    )


def moe_collapse():
    T = 60

    def rich():
        s = [0.160, 0.166, 0.190, 0.158, 0.166, 0.160]
        out = [s]
        for _ in range(T):
            p = [v**1.09 for v in s]
            s = [v / sum(p) for v in p]
            out.append(s)
        return out

    def even():
        out = []
        for t in range(T + 1):
            s = [1 / 6 + 0.02 * math.sin(0.25 * t + 1.7 * i) for i in range(6)]
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
        "The expert that starts a little better gets more images, learns faster from them, and gets even more. A balancing loss grows the more uneven the shares are, and so spreads the images again.",
        "Collapse: Shazeer et al. (2017) · balancing loss: Fedus, Zoph &amp; Shazeer (2022), Switch Transformers",
        mk="schematic",
        h=545,
    )


def moe_speed():
    bx, bw, bh = 520, 900, 54
    rows = [
        (
            "parameters used",
            "per image",
            3795,
            3929,
            "3,795",
            "3,929",
            3929,
            "the same",
        ),
        ("time", "per image", 4.0, 28.2, "4 µs", "28 µs", 28.2, "7× slower"),
    ]
    s = ""
    for r, (n1, n2, d, m, dl, ml, top, fac) in enumerate(rows):
        y = 40 + r * 210
        g = txt(bx - 30, y + 48, n1, 34, INK, "end", 500) + txt(
            bx - 30, y + 90, n2, 28, MUT, "end"
        )
        g += line(bx, y - 10, bx, y + 2 * bh + 22, MUT, 2.5)
        g += rect(bx, y, bw * d / top, bh, P) + txt(
            bx + bw * d / top + 16, y + 38, dl, 30, P, weight=600
        )
        g += rect(bx, y + bh + 12, bw * m / top, bh, V) + txt(
            bx + bw * m / top + 16, y + bh + 50, ml, 30, V, weight=600
        )
        g += txt(1700, y + bh + 20, fac, 36, V if r else MUT, "end", 700)
        if r == 0:
            g += txt(bx + 16, y + 37, "dense head", 26, "#ffffff", weight=600) + txt(
                bx + 16, y + bh + 49, "mixture of experts", 26, "#ffffff", weight=600
            )
        s += g if r == 0 else build(g)
    return slide(
        "moe-speed",
        2,
        ["3.5"],
        'Fewer parameters used: is it <span class="v">faster</span>?',
        s,
        "The mixture uses as many parameters per image as the dense head, and takes about seven times as long. Sending each image to its expert and collecting the answers costs time that no parameter count shows.",
        "Notebook 3.2.6 · one CPU thread, one image at a time",
        h=440,
    )


# ---------------------------------------------------------------- part 3
BAND = (0.44, 0.56)


def panel(x0, y0, w, h, title, groups, rug=True, good_y=32):
    """groups: list of (points, kind); kind in grey, hollow, p, v."""
    g = txt(x0, y0 - 18, title, 30, INK, weight=500)
    g += rect(x0 + w * BAND[0], y0, w * (BAND[1] - BAND[0]), h, PT, op=0.7)
    g += txt(
        x0 + w * (BAND[0] + BAND[1]) / 2, y0 + good_y, "good", 24, P, "middle", 600
    )
    g += rect(x0, y0, w, h, "none", MUT, 2.5)
    g += txt(
        x0 + w,
        y0 + h + (86 if rug else 40),
        "learning rate: matters a lot →",
        24,
        MUT,
        "end",
    )
    g += txt(x0, y0 + h + (86 if rug else 40), "↑ an axis that matters little", 24, MUT)
    style = {"grey": (GREY, GREY), "hollow": (BG, GREY), "p": (P, P), "v": (V, V)}
    for pts, kind in groups:
        f, st = style[kind]
        for u, v in pts:
            g += circ(x0 + w * u, y0 + h * (1 - v), 12, f, st, 4)
            if rug:
                hit = BAND[0] <= u <= BAND[1]
                g += line(
                    x0 + w * u,
                    y0 + h + 14,
                    x0 + w * u,
                    y0 + h + 44,
                    V if hit else MUT,
                    5,
                )
    return g


def trials():
    rnd = random.Random(11)
    rand = [(rnd.uniform(0.04, 0.96), rnd.uniform(0.08, 0.9)) for _ in range(16)]
    grid = [(0.125 + 0.25 * i, 0.14 + 0.24 * j) for i in range(4) for j in range(4)]
    first = rand[:6]
    centre = min(first, key=lambda p: abs(p[0] - 0.5))[0]
    tpe = []
    for i in range(10):
        centre += (0.5 - centre) * 0.45
        tpe.append((centre + rnd.gauss(0, 0.05 - 0.003 * i), rnd.uniform(0.08, 0.9)))
    pop1 = [
        (0.08, 0.3),
        (0.2, 0.75),
        (0.36, 0.45),
        (0.64, 0.6),
        (0.8, 0.25),
        (0.92, 0.7),
    ]
    parents = [pop1[2], pop1[3]]
    gen2 = [(0.29, 0.2), (0.42, 0.74), (0.455, 0.3), (0.575, 0.84), (0.71, 0.42)]
    par2 = [gen2[2], gen2[3]]
    gen3 = [(0.49, 0.13), (0.52, 0.44), (0.47, 0.56), (0.535, 0.7), (0.5, 0.9)]
    return grid, rand, first, tpe, parents, gen2, par2, gen3, pop1


def grid_random():
    grid, rand, *_ = trials()
    kinds = lambda pts: [
        ([p for p in pts if not BAND[0] <= p[0] <= BAND[1]], "grey"),
        ([p for p in pts if BAND[0] <= p[0] <= BAND[1]], "v"),
    ]
    a = panel(90, 60, 700, 340, "grid: 16 trials, 4 learning rates", kinds(grid))
    b = panel(930, 60, 700, 340, "random: 16 trials, 16 learning rates", kinds(rand))
    return slide(
        "grid-random",
        3,
        ["3.6"],
        'Grid and random: where do <span class="v">16 trials</span> go?',
        a + build(b),
        "The ticks under each box are the learning rates that were tried. When one axis matters most, random search tries far more values of it with the same budget.",
        "After Bergstra &amp; Bengio (2012), Random search for hyper-parameter optimization.",
        mk="schematic",
        h=500,
    )


def tpe_slide():
    _, _, first, tpe, *_ = trials()
    x0, y0, w, h = 310, 50, 1100, 380
    a = panel(x0, y0, w, h, "", [(first, "hollow")], rug=False, good_y=-14)
    b = panel(x0, y0, w, h, "", [(first, "hollow"), (tpe, "v")], rug=False, good_y=-14)
    leg = (
        circ(1460, 120, 12, BG, GREY, 4)
        + txt(1484, 130, "first trials", 26, MUT)
        + build(
            circ(1460, 180, 12, V, V, 4)
            + txt(1484, 190, "later trials", 26, V, weight=600)
        )
    )
    return slide(
        "tpe",
        3,
        ["3.6"],
        'TPE: the next trial goes where <span class="v">earlier trials did well</span>',
        a
        + leg.split("<g data-build-in>")[0]
        + build(
            b
            + circ(1460, 180, 12, V, V, 4)
            + txt(1484, 190, "later trials", 26, V, weight=600)
        ),
        "TPE splits the trials so far into good and rest, and draws the next one where the good ones lie. Random search never looks back.",
        mk="schematic",
        h=490,
    )


def population_slide():
    _, _, _, _, parents, gen2, par2, gen3, first = trials()
    x0, y0, w, h = 310, 50, 1100, 380
    XY = lambda p: (x0 + w * p[0], y0 + h * (1 - p[1]))
    a = panel(x0, y0, w, h, "", [(first, "hollow")], rug=False, good_y=-14)
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
    leg = circ(1460, 120, 12, BG, GREY, 4) + txt(1484, 130, "generation 1", 26, MUT)
    return slide(
        "population",
        3,
        ["3.6"],
        'A population: keep the best, <span class="v">breed</span> the next from them',
        a
        + leg
        + build(
            b
            + circ(1460, 180, 12, P, P, 4)
            + txt(1484, 190, "generation 2", 26, P, weight=600)
        )
        + build(
            c
            + circ(1460, 240, 12, V, V, 4)
            + txt(1484, 250, "generation 3", 26, V, weight=600)
        ),
        "The ringed trials are kept; each new generation is made from small changes to them. Good candidates stay in the search instead of being drawn once.",
        mk="schematic",
        h=490,
    )


def space_grows():
    x0, y0, sc, budget = 330, 480, 0.59, 27
    s = line(x0 - 40, y0, 1500, y0, MUT, 2.5)
    for i in range(6):
        n = 3 ** (i + 1)
        x = x0 + i * 180
        covered = min(n, budget)
        g = rect(x, y0 - covered * sc, 110, covered * sc, P)
        if n > budget:
            g += rect(x, y0 - n * sc, 110, (n - budget) * sc, VT, V, 3)
        g += txt(
            x + 55,
            y0 - max(n, budget) * sc - 16,
            f"{n:,}",
            32,
            V if n > budget else P,
            "middle",
            600,
        )
        g += txt(
            x + 55,
            y0 + 40,
            f"{i + 1} ax{'i' if i == 0 else 'e'}s",
            28,
            INK,
            "middle",
            500,
        )
        frac = "all tried" if n <= budget else f"1 in {n // budget}"
        g += txt(x + 55, y0 + 78, frac, 26, MUT, "middle")
        s += g if i < 3 else build(g)
    s += line(x0 - 40, y0 - budget * sc, 1500, y0 - budget * sc, INK, 3, "12 8") + txt(
        1516, y0 - budget * sc + 9, "budget: 27 trials", 28, INK, weight=500
    )
    s += txt(x0 - 60, 60, "combinations ↑", 28, MUT, "end")
    return slide(
        "space-grows",
        3,
        ["3.7"],
        'Every axis <span class="v">multiplies</span> the search space',
        s,
        "With three values per axis, every new axis triples the number of combinations. The budget stays where it was.",
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
        'Successive halving: many trials <span class="v">short</span>, the best ones long',
        s + build(b),
        "Both pictures cost 63 epochs. Halving starts 27 trials and lets one in three pass each rung; training every trial to the end pays for 7.",
        h=570,
    )


def slow_starter():
    rows = json.loads((RESULTS / "search_budget.json").read_text())["tpe"]
    curves = [r["curve"] for r in rows]
    win = max(range(len(curves)), key=lambda i: curves[i][-1])
    kept = max(
        (i for i in range(len(curves)) if curves[i][0] >= 0.85),
        key=lambda i: curves[i][-1],
    )
    x0, y0, w, h = 330, 480, 1050, 430
    X = lambda e: x0 + w * (e - 1) / 19
    Y = lambda v: y0 - h * (max(v, 0.5) - 0.5) / 0.43
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
    for v in (0.6, 0.7, 0.8, 0.9):
        s += line(x0 - 10, Y(v), x0, Y(v), MUT, 2) + txt(
            x0 - 20, Y(v) + 9, f"{v}", 26, MUT, "end"
        )
    for i, c in enumerate(curves):
        if i not in (win, kept):
            s += poly([(X(e + 1), Y(v)) for e, v in enumerate(c)], GREY, 2, op=0.35)
    s += poly([(X(e + 1), Y(v)) for e, v in enumerate(curves[kept])], P, 5)
    s += txt(X(20) + 20, Y(curves[kept][-1]) + 34, "best trial kept", 28, P, weight=600)
    b = line(X(1), y0, X(1), y0 - h, V, 3, "10 8") + txt(
        X(1) + 14, y0 - h + 6, "first rung", 28, V, weight=600
    )
    b += poly([(X(e + 1), Y(v)) for e, v in enumerate(curves[win])], V, 5, "12 8")
    b += circ(X(1), Y(curves[win][0]), 15, VT, V, 5)
    b += txt(
        X(1) + 30,
        Y(curves[win][0]) + 44,
        f"stopped here, at {curves[win][0]:.2f}",
        28,
        V,
        weight=600,
    )
    b += txt(
        X(20) + 20,
        Y(curves[win][-1]) - 8,
        f"would end highest: {curves[win][-1]:.2f}",
        28,
        V,
        weight=600,
    )
    return slide(
        "slow-starter",
        3,
        ["3.7"],
        'The pitfall: a <span class="v">slow starter</span> is stopped',
        s + build(b),
        "Halving ranks trials by how far they get in the first epochs. The trial that ends highest here is far down the pack after one epoch, and the first rung stops it.",
        "The 60 trials of the TPE search, replayed · notebook 3.3.5",
        h=570,
    )


def lr_valley():
    def valley(cx, us, col, title):
        Y = lambda u: 440 - 380 * u * u
        X = lambda u: cx + 330 * u
        g = txt(cx, 34, title, 30, INK, "middle", 500)
        g += poly([(X(-1 + i / 30), Y(-1 + i / 30)) for i in range(61)], MUT, 4)
        for a, b in itertools.pairwise(us):
            g += line(X(a), Y(a), X(b), Y(b), col, 4, marker="v" if col == V else "g")
        g += "".join(circ(X(u), Y(u), 10, col) for u in us)
        return g

    a = valley(
        430, [0.92, -0.88, 0.84, -0.8, 0.77], MUT, "the same big step every time"
    )
    b = valley(
        1290, [0.92, -0.62, 0.38, -0.2, 0.09, -0.03, 0.0], V, "steps that shrink"
    )
    return slide(
        "lr-valley",
        3,
        ["3.8"],
        'Why <span class="v">lower</span> the learning rate during a training?',
        a + build(b),
        "Big steps cross the valley fast and keep overshooting; small steps settle. A scheduler starts with big steps and ends with small ones, inside one training.",
        mk="schematic",
        h=480,
    )


def schedule_or_race():
    x0, y0, w, h = 140, 420, 640, 320
    X = lambda u: x0 + w * u
    Y = lambda v: y0 - h * v
    s = txt(x0, 34, "a scheduler: one training, the step changes", 30, INK, weight=500)
    s += line(x0, y0, x0 + w + 30, y0, MUT, 2.5, marker="g") + line(
        x0, y0, x0, y0 - h - 20, MUT, 2.5, marker="g"
    )
    s += txt(x0 + w + 30, y0 + 44, "epochs →", 26, MUT, "end") + txt(
        x0 + 16, y0 - h - 24, "learning rate ↑", 26, MUT
    )
    s += poly([(X(0), Y(0.9)), (X(1), Y(0.9))], GREY, 5) + txt(
        X(1) + 14, Y(0.9) + 9, "constant", 26, MUT, weight=500
    )
    s += poly(
        [
            (X(0), Y(0.9)),
            (X(0.33), Y(0.9)),
            (X(0.33), Y(0.6)),
            (X(0.66), Y(0.6)),
            (X(0.66), Y(0.3)),
            (X(1), Y(0.3)),
        ],
        P,
        5,
    )
    s += txt(X(1) + 14, Y(0.3) + 9, "step", 26, P, weight=600)
    s += poly(
        [
            (X(i / 40), Y(0.06 + 0.84 * (1 + math.cos(math.pi * i / 40)) / 2))
            for i in range(41)
        ],
        V,
        6,
    )
    s += txt(X(1) + 14, Y(0.06) + 9, "cosine", 26, V, weight=600)
    rx = 1040
    RX = lambda e: rx + 560 * (e - 1) / 19
    b = txt(rx, 34, "a race: whole trials stop", 30, INK, weight=500)
    b += line(rx, y0, rx + 590, y0, MUT, 2.5, marker="g") + line(
        rx, y0, rx, y0 - h - 20, MUT, 2.5, marker="g"
    )
    b += txt(rx + 590, y0 + 44, "epochs →", 26, MUT, "end") + txt(
        rx + 16, y0 - h - 24, "accuracy ↑", 26, MUT
    )
    rnd = random.Random(5)
    for i in range(9):
        end = 20 if i == 0 else (9 if i < 3 else 3)
        top = 0.9 - 0.085 * i + rnd.uniform(-0.015, 0.015)
        pts = [(RX(e), Y(top * (1 - math.exp(-e / 2.5)))) for e in range(1, end + 1)]
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


# ---------------------------------------------------------------- part 4: the admissions detour
def admissions_goals():
    size, pitch, x0 = 54, 64, 360

    def seats(y, name, target, grads_t, grads_o):
        g = txt(x0 - 30, y + 28, name[0], 32, INK, "end", 500) + txt(
            x0 - 30, y + 66, name[1], 32, INK, "end", 500
        )
        for i in range(20):
            is_t = i < target
            grad = i < grads_t if is_t else (i - target) < grads_o
            col, tint = (V, VT) if is_t else (P, PT)
            g += rect(x0 + i * pitch, y, size, size, col if grad else tint, col, 4)
        g += txt(
            x0, y + size + 46, f"{target} from the target group", 30, V, weight=600
        )
        g += txt(
            x0 + 480,
            y + size + 46,
            f"{grads_t + grads_o} will graduate",
            30,
            INK,
            weight=600,
        )
        return g

    s = seats(40, ("places go to the", "highest test scores"), 3, 2, 14)
    b = seats(250, ("more places for", "the target group"), 8, 5, 10)
    leg = rect(x0, 462, 30, 30, V, V, 4) + txt(x0 + 44, 487, "target group", 26, MUT)
    leg += rect(x0 + 260, 462, 30, 30, P, P, 4) + txt(
        x0 + 304, 487, "other applicants", 26, MUT
    )
    leg += rect(x0 + 590, 462, 30, 30, GREY, GREY, 4) + txt(
        x0 + 634, 487, "graduates", 26, MUT
    )
    leg += rect(x0 + 820, 462, 30, 30, GREYT, GREY, 4) + txt(
        x0 + 864, 487, "does not graduate", 26, MUT
    )
    return slide(
        "admissions-goals",
        4,
        ["3.9"],
        'A detour: 20 places at a university, and <span class="v">two goals</span>',
        s + build(b) + leg,
        "An admissions committee wants students who graduate, and more students from a group that is underrepresented. Every way of filling the places is a policy with its own two numbers.",
        mk="made-up numbers",
        h=510,
    )


def admissions_front():
    x0, y0, w, h = 330, 480, 1000, 430
    s = line(x0, y0, x0 + w + 40, y0, MUT, 2.5, marker="g") + line(
        x0, y0, x0, y0 - h - 30, MUT, 2.5, marker="g"
    )
    s += txt(
        x0 + w + 40, y0 + 50, "admitted from the target group →", 28, MUT, "end"
    ) + txt(x0 - 20, y0 - h - 10, "graduates ↑", 28, MUT, "end")
    F = lambda u: 0.86 - 0.62 * u**2.4
    X = lambda u: x0 + w * (0.08 + 0.86 * u)
    Y = lambda v: y0 - h * v
    us = [i / 50 for i in range(51)]
    s += poly([(X(u), Y(F(u))) for u in us], P, 5) + txt(
        X(1) + 20, Y(F(1)) + 40, "Pareto front", 30, P, weight=600
    )
    for u, lab in (
        (0.0, "highest test scores"),
        (0.94, "more places for the target group"),
    ):
        s += circ(X(u), Y(F(u)), 14, PT, P, 5) + txt(
            X(u) + 26, Y(F(u)) + (52 if u == 0 else 6), lab, 26, P, weight=500
        )
    ru, rv = 0.3, 0.5
    lo = next(u for u in us if F(u) < rv)
    b = rect(
        X(ru), Y(F(ru)), X(lo) - X(ru), Y(rv) - Y(F(ru)), VT, V, 2, dash="8 6", op=0.5
    )
    b += poly([(X(u), Y(F(u))) for u in us if ru <= u <= lo], V, 10)
    b += circ(X(ru), Y(rv), 15, BG, INK, 5) + txt(
        X(ru) + 28, Y(rv) + 50, "policy under a fairness rule", 28, INK, weight=600
    )
    b += txt(
        X(ru) + 20,
        Y(F(ru)) - 20,
        "more of the target group and more graduates",
        28,
        V,
        weight=600,
    )
    return slide(
        "admissions-front",
        4,
        ["3.9"],
        'A fairness rule can land <span class="v">inside</span> the front',
        s + build(b),
        "Nilforoshan et al. (2022) prove this for common causal definitions of fairness: almost always, another policy admits more students of the target group and has more graduates. Which point on the front is right stays a choice of values; no formula makes it.",
        "Schematic after Nilforoshan, Gaebler, Shroff &amp; Goel (2022), Causal conceptions of fairness and their consequences, Figure 2 · their applicants are simulated.",
        h=540,
    )


# ---------------------------------------------------------------- part 5
def mix7_bars():
    rows = [
        ("dishes", "Food-101", 75750, 101),
        ("traffic signs", "GTSRB", 26640, 43),
        ("satellite tiles", "EuroSAT", 21600, 10),
        ("galaxies", "Galaxy10", 15962, 10),
        ("blood cells", "BloodMNIST", 13671, 8),
        ("flowers", "Oxford Flowers", 7169, 102),
        ("textures", "DTD", 3760, 47),
    ]
    s = txt(480, 30, "training images", 26, MUT) + txt(
        1700, 30, "classes", 26, MUT, "end"
    )
    for i, (kind, name, n, k) in enumerate(rows):
        y = 50 + i * 66
        s += txt(450, y + 30, kind, 32, INK, "end", 500) + txt(
            450 - 250, y + 30, name, 24, MUT, "end"
        )
        s += rect(480, y, 1000 * n / 75750, 44, P) + txt(
            480 + 1000 * n / 75750 + 16, y + 32, f"{n:,}", 28, P, weight=600
        )
        s += txt(1700, y + 32, str(k), 30, V, "end", 600)
    return s


def mix7():
    return slide(
        "mix7",
        5,
        ["3.12"],
        'The data: <span class="v">seven kinds</span> of image, 321 classes',
        mix7_bars(),
        "One model reads all seven. Every image ran once through a pretrained encoder; your model reads the stored vectors.",
        h=520,
    )


def cost_of_an_image():
    s = rect(60, 190, 150, 150, PT, P, 3) + txt(135, 380, "image", 28, MUT, "middle")
    s += line(220, 265, 290, 265, MUT, 4, marker="g")
    s += rect(300, 60, 760, 410, GREYT, GREY, 3) + txt(
        680, 240, "encoder", 44, MUT, "middle", 500
    )
    s += txt(680, 290, "billions of FLOPs per image", 30, MUT, "middle") + txt(
        680, 334, "ran once, vectors stored", 30, MUT, "middle"
    )
    s += line(1070, 265, 1140, 265, MUT, 4, marker="g")
    s += rect(1150, 150, 36, 230, PT, P, 3) + txt(
        1168, 420, "384", 26, P, "middle", mono=True
    )
    b = line(1196, 265, 1266, 265, MUT, 4, marker="g") + rect(
        1276, 232, 90, 66, VT, V, 4
    )
    b += txt(1321, 200, "head", 34, V, "middle", 600)
    b += txt(1400, 256, "runs in every trial:", 30, V, weight=500) + txt(
        1400, 296, "the cost that differs", 30, V, weight=500
    )
    b += txt(1276, 420, "Linear(384, 64)", 26, INK, mono=True) + txt(
        1276, 458, "2 × 384 × 64 = 49,152 FLOPs", 26, INK, mono=True
    )
    return slide(
        "cost-of-an-image",
        5,
        ["3.11"],
        'What does one image <span class="v">cost</span>?',
        s + build(b),
        "FLOPs per image count the arithmetic of one forward pass: a multiply and an add per weight. The encoder ran once, so the head is the cost that differs between your runs.",
        "Notebook 3.5, “What one image costs”",
        h=500,
    )


def control_in_space():
    x0, y0, w, h = 760, 440, 800, 380
    s = line(x0, y0, x0 + w + 40, y0, MUT, 2.5, marker="g") + line(
        x0, y0, x0, y0 - h - 20, MUT, 2.5, marker="g"
    )
    s += txt(x0 + w + 40, y0 + 46, "FLOPs per image (log) →", 26, MUT, "end") + txt(
        x0 - 16, y0 - h - 4, "accuracy ↑", 26, MUT, "end"
    )
    base = lambda u: 0.08 + 0.5 * (1 - math.exp(-2.4 * u))
    us = [0.08, 0.2, 0.33, 0.46, 0.6, 0.74, 0.9]
    X = lambda u: x0 + w * u
    Y = lambda v: y0 - h * v
    s += "".join(circ(X(u), Y(base(u)), 13, BG, P, 5) for u in us)
    s += circ(x0 + 50, y0 - h + 10, 13, BG, P, 5) + txt(
        x0 + 76, y0 - h + 20, "without your option: the control", 28, P, weight=600
    )
    b = "".join(
        line(X(u), Y(base(u)) - 16, X(u), Y(base(u) + d) + 18, V, 3, "6 5")
        + circ(X(u), Y(base(u) + d), 13, V, V, 4)
        for u, d in zip(us[:5], (0.2, 0.17, 0.13, 0.1, 0.07))
    )
    b += circ(x0 + 50, y0 - h + 58, 13, V, V, 4) + txt(
        x0 + 76, y0 - h + 68, "with your option", 28, V, weight=600
    )
    code = (
        '<foreignObject x="40" y="110" width="640" height="230"><pre xmlns="http://www.w3.org/1999/xhtml" class="code" style="font-size:26px">'
        'SPACE = {\n  "width": Int(8, 64, log=True),\n  <span class="r">"moe": Choice([False, True])</span>,\n}</pre></foreignObject>'
    )
    return slide(
        "control-in-space",
        5,
        ["3.11"],
        'The control is <span class="v">part of the search</span>',
        code + s + build(b),
        "Search your option as a choice beside the width: the trials without it are the control. Your option pays where its trials lie above the others at the same FLOPs.",
        mk="schematic",
        h=500,
    )


def class_study():
    steps = [
        ("idea", "what you expect"),
        ("design", "designs.py"),
        ("study", "make study LESSON=3"),
        ("share", "make share LESSON=3"),
        ("class", "what scale bought"),
    ]
    s = ""
    for i, (name, sub) in enumerate(steps):
        x = 40 + i * 340
        s += rect(x, 120, 280, 150, VT if i == 4 else BG, V if i == 4 else P, 4)
        s += txt(x + 140, 190, name, 40, V if i == 4 else P, "middle", 600) + txt(
            x + 140, 236, sub, 22, MUT, "middle", mono=True
        )
        if i < 4:
            s += line(x + 286, 195, x + 332, 195, MUT, 4, marker="g")
    s += poly([(1540, 276), (1540, 360), (180, 360), (180, 284)], V, 4) + line(
        180, 300, 180, 280, V, 4, marker="v"
    )
    s += txt(
        860,
        404,
        "build on an idea of the class, and name your source",
        30,
        V,
        "middle",
        500,
    )
    return slide(
        "class-study",
        5,
        ["3.12"],
        'Your study, and <span class="v">the class</span>',
        s,
        "One idea, a design that can test it, and a search with its control of equal compute. The next lesson opens on what scale and architecture bought the class.",
        h=440,
    )


# ---------------------------------------------------------------- assemble
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
    out += claim_or_size() + law_loglog() + law_price() + law_range() + grow_together()
    out += taken(
        "lesson3-scaling.html",
        "data-for-parameters",
        1,
        ["3.3"],
        mk="alternative for the previous slide",
    )
    out += taken("lesson3-scaling.html", "gopher", 1, ["3.3"])
    out += compute_area()
    out += taken("lesson3-scaling.html", "chinchilla", 1, ["3.3"])
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
    out += moe_control() + moe_split() + moe_collapse() + moe_speed()
    out += divider(
        "p03",
        3,
        "03 search",
        'Where do the trials go, and <span class="v">how long</span> does each one train?',
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
        "04 trade-offs",
        'Which model is <span class="v">best</span>, if you care about two things?',
        [
            ('<span class="v">Beaten</span>', "Another model is better on both."),
            ('The <span class="v">front</span>', "The models nothing beats."),
            (
                'A <span class="v">detour</span>',
                "The same reading for a fairness rule.",
            ),
        ],
    )
    for sid in ("pareto-beaten", "pareto-front"):
        out += taken(
            "lesson3-tradeoffs.html",
            sid,
            4,
            ["3.9", "3.10"] if sid == "pareto-front" else ["3.9"],
        )
    out += admissions_goals() + admissions_front()
    out += divider(
        "p05",
        5,
        "05 your study",
        'What does an image <span class="v">cost</span>, and what does it buy?',
        [
            ('The <span class="v">mix</span>', "Seven kinds of image, one model."),
            ('The <span class="v">cost</span>', "FLOPs per image, on every trial."),
            (
                'The <span class="v">class</span>',
                "Publish your runs, build on each other.",
            ),
        ],
    )
    out += mix7() + cost_of_an_image() + control_in_space() + class_study()
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
