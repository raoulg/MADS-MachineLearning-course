"""The learning goals of a lesson as a page with one picture per goal.

Reads the goals from references/leerdoelen/les<N>.md and writes
presentations/leerdoelen-les<N>.html, which GitHub Pages publishes. Each goal is a card
with its code, a short title, one inline SVG and the goal text, and has the anchor
#g<N>-<k> so a slide or notebook can link to one picture.

    uv run scripts/goal_pictures.py --lesson 2

The visual grammar: an axis is a row of cells; petrol is the data (a tensor and its axes),
vermilion is the operation acting on it. PICTURES holds the drawing of each goal, in a
400 x 230 viewBox; a lesson is added by adding its goals there.
"""

import argparse
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P, V, INK, MUT, RULE, SOFT, BG = (
    "#1d5a73",
    "#c43a14",
    "#111111",
    "#5a544d",
    "#cfc7bb",
    "#ebe6de",
    "#f4f1ec",
)
PT, VT = "#c9dde4", "#f3cdbf"


def cell(
    x: float, y: float, s: float = 22, fill: str = BG, stroke: str = P, sw: float = 1.5
) -> str:
    return f'<rect x="{x}" y="{y}" width="{s}" height="{s}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def row(
    x: float,
    y: float,
    n: int,
    s: float = 22,
    fills: dict[int, str] | None = None,
    stroke: str = P,
    gap: float = 0,
) -> str:
    fills = fills or {}
    return "".join(
        cell(x + i * (s + gap), y, s, fills.get(i, BG), stroke) for i in range(n)
    )


def t(x: float, y: float, s: str, cls: str = "l", anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{html.escape(s)}</text>'


def arrow(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    c: str = "g",
    sw: float = 1.5,
    dash: str = "",
) -> str:
    col = {"g": MUT, "v": V, "p": P}[c]
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{sw}" marker-end="url(#a{c})"{d}/>'


def path(
    d: str, c: str = "g", sw: float = 1.5, end: bool = True, fill: str = "none"
) -> str:
    col = {"g": MUT, "v": V, "p": P}[c]
    m = f' marker-end="url(#a{c})"' if end else ""
    return f'<path d="{d}" fill="{fill}" stroke="{col}" stroke-width="{sw}"{m}/>'


def box(
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    c: str = "g",
    fill: str = BG,
    cls: str = "l",
) -> str:
    col = {"g": RULE, "v": V, "p": P}[c]
    sw = 1 if c == "g" else 2
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{col}" stroke-width="{sw}"/>'
        + t(x + w / 2, y + h / 2 + 4.5, label, cls, "middle")
    )


PICTURES: dict[str, tuple[str, str]] = {}

# 2.1 skip
PICTURES["2.1"] = (
    "Skip: de route eromheen",
    t(20, 118, "(B, D)", "m")
    + arrow(72, 114, 128, 114)
    + box(132, 92, 110, 44, "blok", "p")
    + arrow(244, 114, 292, 114)
    + f'<circle cx="308" cy="114" r="14" fill="{BG}" stroke="{V}" stroke-width="2"/>'
    + t(308, 119.5, "+", "big v", "middle")
    + arrow(324, 114, 352, 114)
    + t(356, 118, "(B, D)", "m")
    + path("M 100 114 C 100 40, 308 40, 308 96", "v", 2.5)
    + t(204, 48, "skip", "l v", "middle")
    + t(204, 178, "optellen: de shape blijft gelijk", "c", "middle"),
)

# 2.2 kernel
s = 22
x0 = 24
y0 = 62
g = row(x0, y0, 9, s, {2: VT, 3: VT, 4: VT})
g += f'<rect x="{x0 + 2 * s - 3}" y="{y0 - 3}" width="{3 * s + 6}" height="{s + 6}" fill="none" stroke="{V}" stroke-width="2.5"/>'
g += f'<rect x="{x0 + 5 * s - 3}" y="{y0 - 3}" width="{3 * s + 6}" height="{s + 6}" fill="none" stroke="{V}" stroke-width="1.2" stroke-dasharray="4 3"/>'
g += row(x0 + s, y0 + 78, 7, s, {2: V})
for i in (2, 3, 4):
    g += f'<line x1="{x0 + i * s + s / 2}" y1="{y0 + s + 3}" x2="{x0 + 3 * s + s / 2}" y2="{y0 + 78}" stroke="{V}" stroke-width="1.5"/>'
g += (
    t(x0, y0 - 14, "reeks  (B, L, C)", "m")
    + t(x0 + 5 * s + 18, y0 + 50, "zelfde", "l v")
    + t(x0 + 5 * s + 18, y0 + 66, "gewichten", "l v")
)
gx = 268
gy = 50
for r in range(5):
    g += row(gx, gy + r * s, 5, s, {c: VT for c in (1, 2, 3)} if r in (1, 2, 3) else {})
g += f'<rect x="{gx + s - 3}" y="{gy + s - 3}" width="{3 * s + 6}" height="{3 * s + 6}" fill="none" stroke="{V}" stroke-width="2.5"/>'
g += t(gx, gy - 12, "beeld  (B, C, H, W)", "m") + t(
    200, 208, "lokaal scannen: één operatie, twee ranks", "c", "middle"
)
PICTURES["2.2"] = ("Kernel: lokaal scannen", g)

# 2.3 downsampling
s = 20
x0 = 150
g = t(20, 44, "invoer", "m") + row(x0, 30, 8, s)
rows = [
    ("stride 2", "geleerd", 4, "v"),
    ("pooling", "vaste regel", 4, "p"),
    ("global pooling", "as is weg", 1, "p"),
]
for i, (a, b, n, c) in enumerate(rows):
    y = 84 + i * 48
    col = V if c == "v" else P
    g += t(20, y + 10, a, "l v" if c == "v" else "l p") + t(20, y + 24, b, "c")
    w = 2 * s if n == 4 else 8 * s
    for k in range(n):
        g += f'<rect x="{x0 + k * w}" y="{y}" width="{w}" height="{s}" fill="{VT if c == "v" else PT}" stroke="{col}" stroke-width="1.5"/>'
    g += t(x0 + 8 * s + 14, y + 14, f"{n} posities" if n > 1 else "1 vector", "m")
g += arrow(x0 + 4 * s, 54, x0 + 4 * s, 78)
PICTURES["2.3"] = ("Downsampling: minder posities", g)

# 2.4 receptive field
s = 22
x0 = 100
g = (
    row(x0, 150, 9, s, {i: VT for i in range(2, 7)})
    + row(x0 + s, 95, 7, s, {i: VT for i in range(2, 5)})
    + row(x0 + 2 * s, 40, 5, s, {2: V})
)
g += f'<path d="M {x0 + 2 * s} 150 L {x0 + 3 * s} {95 + s} M {x0 + 7 * s} 150 L {x0 + 6 * s} {95 + s} M {x0 + 3 * s} 95 L {x0 + 4 * s} {40 + s} M {x0 + 6 * s} 95 L {x0 + 5 * s} {40 + s}" stroke="{V}" stroke-width="1.5" fill="none"/>'
g += t(20, 55, "laag 2", "m") + t(20, 110, "laag 1", "m") + t(20, 165, "invoer", "m")
g += (
    t(330, 55, "ziet 5", "l v")
    + t(330, 110, "ziet 3", "l v")
    + t(200, 205, "dieper = groter receptive field", "c", "middle")
)
PICTURES["2.4"] = ("Receptive field groeit", g)

# 2.5 state
g = ""
for i in range(5):
    x = 40 + i * 70
    g += (
        cell(x + 9, 30, 22)
        + t(x + 20, 22, f"x{i + 1}", "m", "middle")
        + arrow(x + 20, 54, x + 20, 88)
    )
    op = 0.25 + 0.15 * i
    g += (
        f'<rect x="{x}" y="92" width="40" height="40" fill="{VT}" stroke="{V}" stroke-width="2"/>'
        + t(x + 20, 117, f"h{i + 1}", "m", "middle")
    )
    if i < 4:
        g += arrow(x + 42, 112, x + 66, 112, "v", 2.5)
    g += f'<rect x="{x}" y="150" width="40" height="8" fill="{P}" opacity="{1.0 - 0.2 * i:.2f}"/>'
g += t(20, 180, "wat er nog over is van x1", "c") + t(
    200, 208, "context loopt mee, stap voor stap", "c", "middle"
)
PICTURES["2.5"] = ("State: context meedragen", g)

# 2.6 gate
g = ""
for i, (gv, lab) in enumerate([(0.0, "0  dicht"), (0.5, "0.5"), (1.0, "1  open")]):
    y = 50 + i * 52
    g += f'<line x1="40" y1="{y}" x2="170" y2="{y}" stroke="{P}" stroke-width="10"/>'
    g += (
        f'<circle cx="190" cy="{y}" r="16" fill="{BG}" stroke="{V}" stroke-width="2"/>'
        + t(190, y + 6, "×", "big v", "middle")
    )
    if gv > 0:
        g += f'<line x1="210" y1="{y}" x2="330" y2="{y}" stroke="{P}" stroke-width="{10 * gv}"/>'
    else:
        g += f'<line x1="210" y1="{y}" x2="330" y2="{y}" stroke="{RULE}" stroke-width="1" stroke-dasharray="3 4"/>'
    g += t(342, y + 4, lab, "m v")
g += (
    t(40, 24, "waarde", "m")
    + t(190, 24, "gate", "l v", "middle")
    + t(200, 208, "een geleerde vermenigvuldiger tussen 0 en 1", "c", "middle")
)
PICTURES["2.6"] = ("Gate: wat gaat door", g)

# 2.7 inductive bias
g = ""
cols = [
    (20, "kernel", "lokaliteit"),
    (150, "state", "volgorde"),
    (280, "basis", "periodiciteit"),
]
for x, a, b in cols:
    g += (
        f'<rect x="{x}" y="30" width="110" height="130" fill="{BG}" stroke="{RULE}"/>'
        + t(x + 55, 52, a, "l v", "middle")
        + t(x + 55, 148, b, "l p", "middle")
    )
g += (
    row(35, 85, 4, 20, {1: VT, 2: VT})
    + f'<rect x="52" y="82" width="46" height="26" fill="none" stroke="{V}" stroke-width="2"/>'
)
g += row(165, 85, 4, 20) + arrow(165, 120, 245, 120, "v", 2)
g += f'<path d="M 292 96 q 11 -30 22 0 t 22 0 t 22 0 t 22 0" fill="none" stroke="{V}" stroke-width="2"/>'
g += t(200, 190, "elke bouwsteen is een weddenschap over de data", "c", "middle") + t(
    200, 208, "hij wint alleen als de data zo in elkaar zit", "c", "middle"
)
PICTURES["2.7"] = ("Inductive bias: een weddenschap", g)

# 2.8 1D vs 2D
s = 20
x0 = 80
fills = {5: V, 4: VT, 6: VT, 1: PT, 9: PT}
g = t(20, 44, "1D", "l p") + "".join(
    cell(x0 + i * s, 30, s, fills.get(i, BG))
    + t(x0 + i * s + s / 2, 44, str(i + 1), "n", "middle")
    for i in range(12)
)
gx = 150
gy = 96
g += t(20, gy + 34, "2D", "l p")
for i in range(12):
    r, c = divmod(i, 4)
    g += cell(gx + c * 25, gy + r * 25, 25, fills.get(i, BG)) + t(
        gx + c * 25 + 12.5, gy + r * 25 + 17, str(i + 1), "n", "middle"
    )
g += (
    t(270, gy + 22, "buren naast", "c")
    + f'<rect x="352" y="{gy + 12}" width="12" height="12" fill="{VT}" stroke="{P}"/>'
)
g += (
    t(270, gy + 46, "buren boven/onder", "c")
    + f'<rect x="382" y="{gy + 36}" width="12" height="12" fill="{PT}" stroke="{P}"/>'
)
g += t(200, 208, "zelfde data, andere buren voor de kernel", "c", "middle")
PICTURES["2.8"] = ("1D of 2D: een keuze", g)

# 2.9 encoder or train it yourself
g = (
    t(20, 34, "met encoder", "l p")
    + box(20, 46, 60, 40, "data", "g")
    + arrow(82, 66, 106, 66)
    + box(110, 36, 170, 60, "encoder, bevroren", "p", PT)
    + arrow(282, 66, 306, 66)
    + box(310, 50, 70, 32, "head", "v")
)
g += t(
    200,
    112,
    "weeg af: hoeveel data · afstand tot pretraining · baseline",
    "c",
    "middle",
)
g += (
    t(20, 136, "zelf trainen", "l v")
    + box(20, 146, 60, 40, "data", "g")
    + arrow(82, 166, 106, 166)
    + box(110, 150, 270, 32, "het hele model", "v")
)
g += (
    f'<rect x="300" y="200" width="12" height="12" fill="{BG}" stroke="{V}" stroke-width="2"/>'
    + t(318, 211, "train jij", "c")
)
g += (
    f'<rect x="20" y="200" width="12" height="12" fill="{PT}" stroke="{P}" stroke-width="2"/>'
    + t(38, 211, "iemand anders trainde dit", "c")
)
PICTURES["2.9"] = ("Encoder gebruiken of zelf trainen", g)

# 2.10 design script -> atlas
code = [
    "def narrow(series) -> ...",
    "def wide(series) -> ...",
    "def join(narrow, wide) -> ...",
    "def summarise(joined) -> ...",
]
g = (
    f'<rect x="14" y="26" width="372" height="78" fill="{SOFT}" stroke="{RULE}"/>'
    + "".join(t(24, 44 + i * 17, c, "m") for i, c in enumerate(code))
)
g += arrow(200, 108, 200, 124, "g")
g += (
    box(14, 134, 62, 26, "narrow", "p")
    + box(14, 176, 62, 26, "wide", "p")
    + box(150, 155, 50, 26, "join", "p")
    + box(278, 155, 86, 26, "summarise", "p")
)
g += (
    path("M 78 147 L 148 164", "g")
    + path("M 78 189 L 148 172", "g")
    + arrow(202, 168, 278, 168)
)
g += (
    t(112, 142, "(B,L,C)", "n v", "middle")
    + t(112, 198, "(B,L,C)", "n v", "middle")
    + t(240, 160, "(B,L,2C)", "n v", "middle")
    + arrow(368, 168, 392, 168)
    + t(392, 150, "(B,2C)", "n v", "end")
)
g += t(200, 226, "één getypeerde stap per blok; atlas tekent de flow", "c", "middle")
PICTURES["2.10"] = ("Design script, gelezen in atlas", g)

# 2.11 design and defend
nodes = [
    (200, 40, "verwachting", "v"),
    (330, 110, "ontwerp", "g"),
    (200, 180, "meting", "g"),
    (70, 110, "uitleg", "v"),
]
g = ""
for x, y, lab, c in nodes:
    g += box(x - 52, y - 16, 104, 32, lab, c, VT if c == "v" else BG)
g += (
    path("M 254 46 Q 320 56 330 92", "g")
    + path("M 330 128 Q 320 168 254 178", "g")
    + path("M 146 178 Q 80 168 70 128", "g")
    + path("M 70 92 Q 80 52 146 44", "g")
)
g += (
    f'<rect x="20" y="206" width="12" height="12" fill="{VT}" stroke="{V}" stroke-width="2"/>'
    + t(38, 217, "van jou, zonder agent", "c")
)
g += f'<rect x="220" y="206" width="12" height="12" fill="{BG}" stroke="{RULE}"/>' + t(
    238, 217, "samen met de agent", "c"
)
PICTURES["2.11"] = ("Ontwerpen en verdedigen", g)

# 2.12 class
g = ""
ys = [44, 78, 112, 146]
for i, y in enumerate(ys):
    g += (
        f'<circle cx="34" cy="{y}" r="11" fill="{BG}" stroke="{P}" stroke-width="1.5"/>'
        + arrow(48, y, 128, 92 + (y - 96) * 0.25)
    )
g += box(132, 70, 90, 52, "runs-repo", "p", PT) + arrow(224, 96, 262, 96)
g += (
    f'<rect x="266" y="54" width="114" height="84" fill="{BG}" stroke="{P}" stroke-width="2"/>'
    + t(323, 72, "overzicht", "l p", "middle")
)
for i, h in enumerate([22, 34, 16, 42, 28]):
    g += f'<rect x="{280 + i * 18}" y="{128 - h}" width="12" height="{h}" fill="{V if i == 3 else PT}" stroke="{P}"/>'
g += path("M 323 140 C 323 200, 110 200, 50 164", "v", 2.5) + t(
    200, 172, "bouw voort, noem je bron", "l v", "middle"
)
g += t(20, 22, "13 studenten", "c") + t(
    200, 222, "publiceren → bekijken → voortbouwen", "c", "middle"
)
PICTURES["2.12"] = ("Werken als klas", g)


def dot(x: float, y: float, r: float = 5, fill: str = PT, stroke: str = P) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'


def line(d: str, col: str = P, sw: float = 2, dash: str = "") -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{sw}"{extra}/>'


# 1.1 tensor shapes
def grid(x: float, y: float, rows: int, cols: int, s: float) -> str:
    return "".join(row(x, y + r * s, cols, s) for r in range(rows))


def cube(x: float, y: float, depth: int, s: float) -> str:
    return "".join(
        grid(x + (depth - 1 - d) * 6, y + d * 6, 3, 4, s) for d in range(depth)
    )


g = (
    row(18, 84, 4, 14)
    + t(46, 150, "1D  vector", "l p", "middle")
    + t(46, 168, "(D)", "m", "middle")
)
g += (
    grid(108, 70, 3, 4, 14)
    + t(136, 150, "2D  matrix", "l p", "middle")
    + t(136, 168, "(H, W)", "m", "middle")
)
g += (
    cube(196, 62, 3, 14)
    + t(230, 150, "3D  tensor", "l p", "middle")
    + t(230, 168, "(C, H, W)", "m", "middle")
)
g += (
    cube(296, 66, 3, 9)
    + cube(346, 66, 3, 9)
    + t(340, 150, "4D  batched tensor", "l p", "middle")
    + t(344, 168, "(B, C, H, W)", "m", "middle")
)
g += t(200, 200, "een as van lengte 1 kost niets:", "c", "middle") + t(
    200, 218, "(H, W) = (1, H, W)", "m v", "middle"
)
PICTURES["1.1"] = ("Vector, matrix, tensor", g)

# 1.2 shape and meaning
g = ""
for gx, label, sub in (
    (30, "(3, 1, 4)", "3 items, elk 1 vector"),
    (230, "(1, 3, 4)", "1 item in 3 chunks"),
):
    for r in range(3):
        g += row(gx, 60 + r * 26, 4, 26)
    g += t(gx, 46, label, "m") + t(gx + 52, 162, sub, "l v", "middle")
for r in range(3):
    g += f'<rect x="26" y="{57 + r * 26}" width="112" height="24" rx="3" fill="none" stroke="{V}" stroke-width="2"/>'
g += f'<rect x="225" y="55" width="114" height="88" rx="3" fill="none" stroke="{V}" stroke-width="2"/>'
g += t(
    200,
    205,
    "zelfde getallen: de plek van de 1 zegt wat het is",
    "c",
    "middle",
)
PICTURES["1.2"] = ("Shape is niet betekenis", g)

# 1.3 the encoder has ordered the space
g = ""
for x, y in ((70, 70), (95, 55), (100, 90), (125, 72), (82, 100)):
    g += dot(x, y)
for x, y in ((270, 130), (300, 115), (305, 150), (330, 135), (285, 158)):
    g += f'<rect x="{x - 5}" y="{y - 5}" width="10" height="10" fill="{BG}" stroke="{P}" stroke-width="1.5"/>'
g += dot(150, 100, 7, VT, V) + t(150, 126, "nieuw", "l v", "middle")
g += (
    line("M 144 96 L 130 76", V, 2)
    + t(96, 34, "klasse A", "m", "middle")
    + t(304, 96, "klasse B", "m", "middle")
)
g += t(200, 205, "de dichtste buur geeft het label, zonder training", "c", "middle")
PICTURES["1.3"] = ("De ruimte is al geordend", g)

# 1.4 embed once, train many heads
g = box(20, 60, 120, 70, "encoder", "p", PT) + t(80, 150, "1× duur", "l p", "middle")
g += (
    arrow(142, 95, 176, 95)
    + box(180, 72, 70, 46, "cache", "p")
    + arrow(252, 95, 286, 95)
)
for i in range(5):
    g += f'<rect x="{292 + i * 8}" y="{64 + i * 8}" width="62" height="30" fill="{VT}" stroke="{V}" stroke-width="2"/>'
g += t(355, 118, "head", "l", "middle") + t(338, 150, "100× goedkoop", "l v", "middle")
g += t(200, 205, "een search vermenigvuldigt alleen de head", "c", "middle")
PICTURES["1.4"] = ("Eén keer duur, daarna goedkoop", g)

# 1.5 the encoder moves the score, the head barely
g = (
    t(20, 72, "andere encoder", "l p")
    + f'<rect x="150" y="56" width="170" height="24" fill="{PT}" stroke="{P}" stroke-width="1.5"/>'
)
g += t(328, 73, "tot 14", "m")
g += (
    t(20, 122, "andere head", "l v")
    + f'<rect x="150" y="106" width="16" height="24" fill="{VT}" stroke="{V}" stroke-width="1.5"/>'
)
g += t(174, 123, "zelden meer dan 1", "m")
g += line("M 150 44 L 150 142", MUT, 1) + t(
    150, 162, "verschil in accuracy op DTD, in punten", "c"
)
g += t(200, 205, "besteed je tijd waar de score beweegt", "c", "middle")
PICTURES["1.5"] = ("De encoder beslist", g)

# 1.6 linear collapse and the nonlinearity
g = t(20, 40, "zonder activatie", "l p")
for i in range(3):
    g += box(20 + i * 62, 52, 54, 28, "W", "p")
    if i < 2:
        g += arrow(75 + i * 62, 66, 81 + i * 62, 66)
g += (
    t(214, 71, "=", "big", "middle")
    + box(232, 52, 54, 28, "W", "p")
    + line("M 310 82 L 380 50")
)
g += t(20, 124, "met activatie", "l v")
for i in range(3):
    g += box(20 + i * 62, 136, 54, 28, "σ(W)", "v")
    if i < 2:
        g += arrow(75 + i * 62, 150, 81 + i * 62, 150)
g += arrow(200, 150, 300, 150) + line(
    "M 310 164 L 328 140 L 346 158 L 362 132 L 380 150", V
)
g += t(
    200, 205, "elke unit voegt een knik toe: meer units, meer capaciteit", "c", "middle"
)
PICTURES["1.6"] = ("Zonder nonlinearity geen diepte", g)

# 1.7 one training step
nodes = [
    (200, 36, "batch", "p"),
    (330, 104, "voorspel", "p"),
    (200, 172, "loss", "p"),
    (76, 104, "gradient descent", "v"),
]
g = "".join(
    box(x - 60, y - 16, 120, 32, lab, c, VT if c == "v" else BG)
    for x, y, lab, c in nodes
)
g += (
    path("M 262 42 Q 322 52 330 86")
    + path("M 330 122 Q 322 162 262 170")
    + path("M 138 170 Q 84 160 76 122")
)
g += path("M 76 86 Q 84 48 138 40", "v", 2.5)
g += t(204, 100, "gradient: richting", "c", "middle") + t(
    204, 118, "optimizer: hoe ver", "c", "middle"
)
g += t(200, 216, "de hele lus is één stap, steeds opnieuw", "c", "middle")
PICTURES["1.7"] = ("Eén trainingsstap: de hele lus", g)

# 1.8 loss and metric
g = (
    box(20, 40, 80, 36, "loss", "v", VT)
    + arrow(102, 58, 150, 58)
    + line("M 160 44 C 190 100, 220 20, 250 70", V)
)
g += arrow(262, 58, 296, 58) + t(302, 62, "optimizer", "l")
g += t(160, 96, "glad: differentieerbaar", "c")
g += box(20, 124, 80, 36, "metric", "p", PT) + arrow(102, 142, 150, 142)
g += (
    dot(166, 142)
    + dot(244, 142, 5, BG, P)
    + line("M 174 142 L 236 142", P, 2, "4 3")
    + t(166, 126, "echt", "c", "middle")
    + t(244, 126, "voorspeld", "c", "middle")
    + arrow(262, 142, 296, 142)
    + t(302, 146, "jij", "l")
)
g += t(160, 180, "een afstand die je kunt lezen: MAPE 4 %", "c")
g += t(200, 216, "twee getallen, twee lezers", "c", "middle")
PICTURES["1.8"] = ("Loss is geen metric", g)

# 1.9 overfitting
g = line("M 50 30 L 50 160 L 370 160", MUT, 1) + t(210, 180, "epochs", "c", "middle")
g += line("M 60 50 C 120 120, 220 145, 360 152", P) + t(364, 146, "train", "l p", "end")
g += line("M 60 56 C 120 120, 190 120, 220 112 S 320 78, 360 60", V) + t(
    364, 52, "valid", "l v", "end"
)
g += line("M 205 30 L 205 160", MUT, 1, "4 4") + t(
    205, 24, "vanaf hier overfit", "c", "middle"
)
g += t(200, 212, "regularisatie helpt alleen als de head het plafond is", "c", "middle")
PICTURES["1.9"] = ("Waar train en valid uiteengaan", g)


# 1.10 when a difference is a result
def interval(x: float, y: float, half: float, col: str, fill: str) -> str:
    return line(
        f"M {x} {y - half} L {x} {y + half} M {x - 6} {y - half} L {x + 6} {y - half} M {x - 6} {y + half} L {x + 6} {y + half}",
        col,
    ) + dot(x, y, 5, fill, col)


g = (
    t(100, 34, "geen resultaat", "l", "middle")
    + interval(70, 100, 34, P, PT)
    + interval(130, 88, 34, V, VT)
)
g += (
    t(300, 34, "resultaat", "l v", "middle")
    + interval(270, 124, 16, P, PT)
    + interval(330, 66, 16, V, VT)
)
for x in (70, 270):
    g += t(x, 168, "baseline", "m", "middle") + t(x + 60, 168, "model", "m", "middle")
g += line("M 200 24 L 200 172", RULE, 1)
g += t(
    200, 205, "de marge moet groter zijn dan de spreiding tussen runs", "c", "middle"
)
PICTURES["1.10"] = ("Ruis of resultaat", g)

# 1.11 kinds of learning problems
g = ""
kinds = [
    (20, 24, "classificatie"),
    (210, 24, "regressie"),
    (20, 118, "reinforcement"),
    (210, 118, "unsupervised"),
]
for x, y, lab in kinds:
    g += (
        f'<rect x="{x}" y="{y}" width="170" height="82" fill="{BG}" stroke="{RULE}"/>'
        + t(x + 85, y + 72, lab, "l p", "middle")
    )
g += (
    dot(60, 48)
    + dot(80, 60)
    + dot(130, 44, 5, VT, V)
    + dot(150, 58, 5, VT, V)
    + line("M 105 32 L 105 74", MUT, 1, "4 3")
)
g += "".join(
    dot(x, y, 4) for x, y in ((240, 66), (270, 54), (300, 50), (330, 38))
) + line("M 232 70 L 350 34", V)
g += box(34, 130, 48, 22, "actie", "g", BG, "c") + box(
    108, 130, 66, 22, "beloning", "g", BG, "c"
)
g += path("M 84 141 L 106 141", "v") + path("M 140 154 Q 100 172 60 154", "v")
g += "".join(
    dot(x, y, 4)
    for x, y in ((250, 142), (262, 152), (246, 156), (320, 140), (334, 150), (322, 158))
)
g += t(200, 222, "wat je data je geeft, bepaalt het soort probleem", "c", "middle")
PICTURES["1.11"] = ("Vier soorten leerproblemen", g)

# 1.12 an architecture as a chain of shapes
g = (
    t(14, 68, "(B, 384)", "n v")
    + arrow(66, 64, 86, 64)
    + box(90, 50, 70, 28, "tak A", "p")
)
g += (
    t(14, 138, "(B, 512)", "n v")
    + arrow(66, 134, 86, 134)
    + box(90, 120, 70, 28, "tak B", "p")
)
g += (
    path("M 162 64 L 204 92")
    + path("M 162 134 L 204 106")
    + box(208, 84, 64, 30, "concat", "p")
)
g += (
    arrow(274, 99, 308, 99)
    + box(312, 84, 50, 30, "head", "v", VT)
    + t(337, 136, "(B, K)", "n v", "middle")
)
g += (
    t(184, 58, "(B, H)", "n v", "middle")
    + t(184, 150, "(B, H)", "n v", "middle")
    + t(291, 76, "(B, 2H)", "n v", "middle")
)
g += t(200, 205, "op elke pijl de shape, vóórdat er code draait", "c", "middle")
PICTURES["1.12"] = ("Compositie lezen", g)

# 1.13 the baseline comes first
g = line("M 30 110 L 370 110", MUT, 1) + t(30, 130, "score", "c")
g += line("M 90 96 L 90 124", P, 3) + t(90, 84, "1-NN baseline", "l p", "middle")
g += (
    f'<rect x="168" y="100" width="44" height="20" fill="{PT}" stroke="{P}" stroke-width="1.5"/>'
    + t(190, 84, "lineaire head", "l p", "middle")
)
g += t(190, 140, "± tussen runs", "m", "middle")
g += f'<rect x="236" y="94" width="120" height="32" fill="none" stroke="{V}" stroke-width="2" stroke-dasharray="5 4"/>'
g += t(296, 84, "je search", "l v", "middle") + t(296, 146, "pas daarna", "m", "middle")
g += t(200, 205, "zonder vergelijkingspunt zegt een score niets", "c", "middle")
PICTURES["1.13"] = ("Eerst de baseline", g)

# 1.14 read a search per axis
g = ""
for gx, lab, sub, ys, col in (
    (30, "as A", "doet ertoe", (128, 112, 92, 70, 52), V),
    (220, "as B", "vlak", (92, 98, 90, 96, 93), P),
):
    g += line(f"M {gx} 40 L {gx} 140 L {gx + 150} 140", MUT, 1) + t(
        gx + 75, 30, lab, "l", "middle"
    )
    g += "".join(
        dot(gx + 20 + i * 28, y, 5, VT if col == V else PT, col)
        for i, y in enumerate(ys)
    )
    g += t(gx + 75, 160, sub, "l v" if col == V else "l p", "middle")
g += t(200, 190, "verwachting eerst opschrijven, dan per as lezen", "c", "middle")
g += t(200, 208, "een vlakke as is ook een uitkomst", "c", "middle")
PICTURES["1.14"] = ("Een search per as lezen", g)

PICTURES["1.15"] = ("Publiceren en voortbouwen", PICTURES["2.12"][1])


# 3.1 a gate model beside a smaller baseline, then beside a control of equal size
def l3_model(x: float, y: float, n: int, gate: set[int], s: float = 16) -> str:
    return "".join(
        cell(x + i * s, y, s, VT, V) if i in gate else cell(x + i * s, y, s, PT, P)
        for i in range(n)
    )


g = t(20, 36, "zonder control", "l")
g += t(20, 58, "gate", "m") + l3_model(84, 46, 8, {3, 4}) + t(220, 58, "120k", "m")
g += t(20, 80, "basis", "m") + l3_model(84, 68, 5, set()) + t(220, 80, "75k", "m")
g += t(262, 70, "gate of grootte?", "l v")
g += line("M 20 100 L 380 100", RULE, 1)
g += t(20, 126, "capacity-matched control", "l p")
g += t(20, 148, "gate", "m") + l3_model(84, 136, 8, {3, 4}) + t(220, 148, "120k", "m")
g += t(20, 170, "control", "m") + l3_model(84, 158, 8, set()) + t(220, 170, "120k", "m")
g += t(262, 160, "verschil = gate", "l v")
g += t(200, 205, "zelfde parameters, één verschil", "c", "middle")
PICTURES["3.1"] = ("Vergelijk bij gelijke capaciteit", g)


# 3.2 three measured scaling curves that flatten, the steepest on the data axis
def l3_curve(px: float, h: float, k: float, col: str, fill: str) -> str:
    pts = [(px + 8 + i * 17, 140 - h * (1 - 0.5 ** (k * i))) for i in range(6)]
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    return line(d, col, 1.5) + "".join(dot(x, y, 3.5, fill, col) for x, y in pts)


g = t(20, 32, "score", "c")
for px, lab, h, k, col, fill in (
    (30, "parameters", 45, 1.0, P, PT),
    (155, "data", 80, 0.8, V, VT),
    (280, "rekentijd", 30, 1.5, P, PT),
):
    g += line(f"M {px} 40 L {px} 150 L {px + 100} 150", MUT, 1)
    g += l3_curve(px, h, k, col, fill)
    g += t(px + 50, 168, lab, "c v" if col == V else "c", "middle")
g += t(205, 34, "hier het meest", "c v", "middle")
g += t(200, 205, "gemeten, niet afgeleid: per probleem een andere as", "c", "middle")
PICTURES["3.2"] = ("Scaling curves vlakken af", g)

# 3.3 gate, highway and mixture of experts as three rows with (B, D) in and out
g = ""
for y, lab in ((50, "gate"), (105, "highway"), (165, "MoE")):
    g += t(20, y + 5, lab, "l v") + t(84, y + 4, "(B, D)", "n")
    g += arrow({50: 236, 105: 298, 165: 288}[y], y, 326, y) + t(
        330, y + 4, "(B, D)", "n"
    )
g += arrow(124, 50, 206, 50)
g += f'<circle cx="222" cy="50" r="13" fill="{BG}" stroke="{V}" stroke-width="2"/>'
g += t(222, 56, "×", "big v", "middle")
g += arrow(222, 22, 222, 35, "v") + t(230, 30, "g", "n v")
g += (
    arrow(124, 105, 166, 105)
    + box(170, 93, 70, 24, "blok", "p")
    + arrow(242, 105, 268, 105)
)
g += f'<circle cx="284" cy="105" r="13" fill="{BG}" stroke="{V}" stroke-width="2"/>'
g += t(284, 109, "mix", "n v", "middle")
g += path("M 146 105 C 146 78, 284 78, 284 90", "v", 2) + t(
    215, 76, "skip", "n v", "middle"
)
g += arrow(124, 165, 138, 165) + box(140, 153, 54, 24, "router", "v", VT, "n")
for i, ey in enumerate((141, 159, 177)):
    chosen = i == 0
    g += f'<rect x="216" y="{ey}" width="48" height="14" fill="{PT if chosen else BG}" stroke="{P if chosen else RULE}" stroke-width="1.5"/>'
    g += line(
        f"M 196 165 L 214 {ey + 7}",
        V if chosen else RULE,
        2 if chosen else 1,
        "" if chosen else "3 3",
    )
    g += line(f"M 264 {ey + 7} L 288 165", P if chosen else RULE, 1.5 if chosen else 1)
g += t(200, 214, "steeds fijner: weeg, meng, kies", "c", "middle")
PICTURES["3.3"] = ("Drie resoluties van één idee", g)

# 3.4 four trained experts with one used per input, and the load per expert
g = t(37, 80, "input", "m", "middle") + cell(26, 89, 22, VT, V)
for i in range(4):
    ey = 40 + i * 32
    used = i == 1
    g += f'<rect x="100" y="{ey}" width="70" height="24" fill="{VT if used else PT}" stroke="{V if used else P}" stroke-width="{2 if used else 1.5}"/>'
    if used:
        g += arrow(50, 100, 98, ey + 12, "v", 2)
    else:
        g += line(f"M 50 100 L 98 {ey + 12}", RULE, 1, "3 3")
g += t(100, 182, "getraind: 4 experts", "l p", "middle")
g += t(100, 200, "gebruikt: 1 per input", "l v", "middle")
g += line("M 196 30 L 196 200", RULE, 1)
g += t(298, 36, "load per expert", "c", "middle")
for x0, hs, lab, cls in (
    (214, (96, 6, 4, 4), "collapse", "c v"),
    (306, (26, 24, 27, 25), "met balancing", "c"),
):
    g += line(f"M {x0 - 4} 150 L {x0 + 70} 150", MUT, 1)
    for i, h in enumerate(hs):
        hot = cls == "c v" and i == 0
        g += f'<rect x="{x0 + i * 18}" y="{150 - h}" width="12" height="{h}" fill="{VT if hot else PT}" stroke="{V if hot else P}" stroke-width="1.5"/>'
    g += t(x0 + 33, 168, lab, cls, "middle")
PICTURES["3.4"] = ("Getraind is niet gebruikt", g)

# 3.5 where grid, random, TPE and a population put their trials
L3_RANDOM = [
    (14, 62),
    (30, 20),
    (48, 70),
    (70, 44),
    (22, 40),
    (60, 12),
    (38, 50),
    (74, 70),
    (56, 30),
    (10, 14),
    (66, 58),
    (30, 74),
]
L3_EARLY = [(14, 60), (30, 18), (46, 70), (70, 52), (22, 40), (40, 46)]
L3_LATE = [(56, 22), (64, 26), (60, 16), (52, 30), (68, 20), (62, 32), (58, 26)]
L3_KEPT = [(52, 28), (66, 18), (60, 40), (34, 52)]
g = ""
for k, (name, sub) in enumerate(
    (
        ("grid", "weet niets"),
        ("random", "trekt"),
        ("TPE", "leert"),
        ("populatie", "houdt vast"),
    )
):
    x0, y0 = 18 + k * 94, 40
    g += f'<rect x="{x0}" y="{y0}" width="84" height="84" fill="{BG}" stroke="{RULE}"/>'
    g += f'<circle cx="{x0 + 60}" cy="{y0 + 24}" r="14" fill="{VT}"/>'
    if k == 0:
        g += "".join(
            dot(x0 + 12 + c * 20, y0 + 12 + r * 20, 3)
            for r in range(4)
            for c in range(4)
        )
    elif k == 1:
        g += "".join(dot(x0 + a, y0 + b, 3) for a, b in L3_RANDOM)
    elif k == 2:
        g += "".join(dot(x0 + a, y0 + b, 3, BG, RULE) for a, b in L3_EARLY)
        g += "".join(dot(x0 + a, y0 + b, 3, VT, V) for a, b in L3_LATE)
    else:
        for a, b in L3_KEPT:
            g += dot(x0 + a + 7, y0 + b + 6, 2.5, VT, V) + dot(
                x0 + a - 6, y0 + b + 7, 2.5, VT, V
            )
            g += dot(x0 + a, y0 + b, 4, P, P)
    g += t(x0 + 42, 144, name, "l", "middle") + t(
        x0 + 42, 162, sub, "c v" if k else "c", "middle"
    )
g += arrow(30, 182, 370, 182)
g += t(200, 205, "elke stap onthoudt meer van eerdere trials", "c", "middle")
PICTURES["3.5"] = ("Wat het algoritme weet", g)

# 3.6 ASHA stopping trials at rungs, and combinations per axis against a fixed budget
g = t(110, 32, "ASHA", "l v", "middle")
for x in (80, 120, 160):
    g += line(f"M {x} 37 L {x} 146", V, 1, "3 3")
for i, end in enumerate((196, 120, 160, 80, 80, 120, 80, 80)):
    y = 40 + i * 14
    g += line(f"M 36 {y} L {end} {y}", P, 2)
    if end < 196:
        g += line(f"M {end} {y - 5} L {end} {y + 5}", V, 2.5)
g += t(116, 160, "epochs", "c", "middle")
g += t(110, 178, "slechte trials stoppen vroeg", "c", "middle")
g += t(290, 32, "combinaties", "c", "middle")
g += line("M 232 146 L 380 146", MUT, 1)
for x, n, lab in ((240, 5, "1 as"), (290, 25, "2 assen"), (340, 125, "3 assen")):
    h = n * 0.88
    big = n > 40
    g += f'<rect x="{x}" y="{146 - h:.1f}" width="30" height="{h:.1f}" fill="{VT if big else PT}" stroke="{V if big else P}" stroke-width="1.5"/>'
    g += t(x + 15, 162, str(n), "n", "middle") + t(x + 15, 178, lab, "c", "middle")
g += line("M 232 111 L 382 111", MUT, 1.5, "5 4") + t(232, 105, "budget", "c")
g += t(200, 205, "veel korte trials, alleen de besten lang", "c", "middle")
PICTURES["3.6"] = ("Veel kort, weinig lang", g)


# 3.7 a learning-rate curve from coarse to fine above ASHA's narrowing trials
def l3_smooth(u: float) -> float:
    return 3 * u * u - 2 * u * u * u


g = t(20, 44, "lr", "m") + t(20, 150, "trials", "m")
pts = [(70 + 300 * i / 30, 40 + 58 * l3_smooth(i / 30)) for i in range(31)]
g += line("M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts), V, 2.5)
g += t(90, 92, "kettingzaag", "l v") + t(330, 76, "guts", "l v")
for x in (170, 270):
    g += line(f"M {x} 114 L {x} 186", MUT, 1, "3 3")
for i in range(8):
    end = 370 if i == 0 else 270 if i < 4 else 170
    g += f'<rect x="70" y="{120 + i * 8}" width="{end - 70}" height="5" fill="{PT}" stroke="{P}" stroke-width="1"/>'
g += t(200, 212, "breed en goedkoop eerst, smal en precies daarna", "c", "middle")
PICTURES["3.7"] = ("Eerst grof, dan fijn", g)

# 3.8 trials by latency and score, dot size for parameters, the Pareto front in vermilion
g = line("M 50 30 L 50 170 L 370 170", MUT, 1) + t(56, 36, "score", "c")
for x, y, r in (
    (110, 150, 5),
    (150, 128, 7),
    (200, 108, 6),
    (230, 88, 8),
    (290, 80, 8),
    (320, 120, 5),
    (270, 140, 4),
    (180, 150, 3),
    (360, 100, 7),
):
    g += dot(x, y, r)
front = ((80, 140, 3), (120, 96, 4), (170, 70, 5), (250, 52, 7), (340, 44, 9))
g += line("M " + " L ".join(f"{x} {y}" for x, y, _ in front), V, 2)
g += "".join(dot(x, y, r, VT, V) for x, y, r in front)
g += t(58, 70, "Pareto-front", "l v") + t(330, 33, "hoogste score", "c", "end")
g += t(210, 188, "latency", "c", "middle") + t(
    380, 188, "grootte = parameters", "c", "end"
)
g += t(200, 210, "elk punt op het front is een keuze", "c", "middle")
PICTURES["3.8"] = ("De beste trial bestaat niet", g)

# 3.9 the search touches train and valid many times, one run touches test
g = box(100, 30, 120, 30, "search", "g") + t(94, 49, "50 trials", "n", "end")
g += "".join(arrow(112 + i * 20, 62, 44 + i * 46, 106, "g", 1.2) for i in range(6))
g += box(290, 30, 80, 30, "één run", "v", VT) + arrow(330, 62, 330, 106, "v", 2.5)
g += (
    f'<rect x="30" y="110" width="180" height="28" fill="{PT}" stroke="{P}" stroke-width="1.5"/>'
    + f'<rect x="210" y="110" width="80" height="28" fill="{PT}" stroke="{P}" stroke-width="1.5"/>'
    + f'<rect x="290" y="110" width="80" height="28" fill="{BG}" stroke="{V}" stroke-width="2.5"/>'
)
g += (
    t(120, 129, "train", "l p", "middle")
    + t(250, 129, "valid", "l p", "middle")
    + t(330, 129, "test", "l p", "middle")
)
g += t(160, 158, "de search keek hier", "c", "middle") + t(
    330, 158, "rapporteer dit", "c v", "middle"
)
PICTURES["3.9"] = ("Het getal dat je rapporteert", g)

# 3.10 two capacity curves with run-to-run intervals, one gain, one flat
g = ""
for px, lab, head, cls, col, fill, pts, half in (
    (30, "parameters", "winst > spreiding", "l v", V, VT, (132, 104, 84, 72), 9),
    (220, "data", "binnen de spreiding", "l p", P, PT, (98, 94, 100, 95), 14),
):
    g += line(f"M {px} 44 L {px} 150 L {px + 150} 150", MUT, 1)
    xs = [px + 20 + i * 35 for i in range(4)]
    g += line("M " + " L ".join(f"{x} {y}" for x, y in zip(xs, pts)), col, 1, "4 3")
    g += "".join(interval(x, y, half, col, fill) for x, y in zip(xs, pts))
    g += t(px + 75, 34, head, cls, "middle") + t(px + 75, 168, lab, "c", "middle")
g += t(200, 205, "verticale balk: spreiding tussen runs", "c", "middle")
PICTURES["3.10"] = ("Twee curves, met spreiding", g)

# 3.11 a search plan: axes, algorithm, scheduler, and trials filling one budget
g = t(20, 48, "assen", "l") + t(20, 86, "algoritme", "l")
g += t(20, 124, "scheduler", "l") + t(20, 172, "budget", "l")
g += "".join(
    cell(110 + i * 24, 32, 24, VT if i < 2 else BG, V if i < 2 else P) for i in range(6)
)
g += t(262, 49, "2 van 6 eerst", "c")
g += box(110, 70, 60, 24, "grid", "g", BG, "c") + box(
    174, 70, 60, 24, "random", "g", BG, "c"
)
g += box(238, 70, 60, 24, "TPE", "v", VT, "c")
g += box(110, 108, 60, 24, "geen", "g", BG, "c") + box(
    174, 108, 60, 24, "ASHA", "v", VT, "c"
)
g += f'<rect x="110" y="146" width="144" height="46" fill="none" stroke="{MUT}" stroke-width="1.5" stroke-dasharray="5 4"/>'
for i, end in enumerate((250, 224, 184, 184, 144, 144, 144, 144)):
    g += (
        f'<rect x="114" y="{150 + i * 5}" width="{end - 114}" height="3.5" fill="{P}"/>'
    )
g += t(266, 164, "8 trials", "m") + t(266, 182, "max 9 epochs", "m")
g += t(200, 214, "vooraf gekozen, binnen één budget", "c", "middle")
PICTURES["3.11"] = ("Een search op budget", g)

# 3.12 routing and its control published to the runs-repo, read and built on by the class
g = t(20, 36, "hartslagen", "c")
g += box(20, 46, 100, 30, "routing", "v", VT) + t(128, 65, "120k", "n")
g += box(20, 90, 100, 30, "control", "p") + t(128, 109, "120k", "n")
g += line("M 156 61 L 162 61 L 162 105 L 156 105", MUT, 1.5) + arrow(162, 83, 184, 83)
for x in (200, 228, 256):
    g += f'<circle cx="{x}" cy="28" r="7" fill="{BG}" stroke="{P}" stroke-width="1.5"/>'
    g += arrow(x, 36, x, 60)
g += box(188, 62, 80, 42, "runs-repo", "p", PT) + arrow(270, 83, 290, 83)
g += f'<rect x="294" y="44" width="88" height="78" fill="{BG}" stroke="{P}" stroke-width="2"/>'
g += t(338, 62, "overzicht", "l p", "middle")
for i, h in enumerate((22, 34, 16, 40, 28)):
    g += f'<rect x="{304 + i * 16}" y="{116 - h}" width="10" height="{h}" fill="{V if i == 3 else PT}" stroke="{P}"/>'
g += path("M 338 124 C 338 180, 90 180, 70 124", "v", 2.5)
g += t(204, 194, "bouw voort, noem je bron", "l v", "middle")
g += t(200, 216, "wat leverde die capaciteit op?", "c", "middle")
PICTURES["3.12"] = ("Routing, met control, als klas", g)

# 4.1 attention: one output position as a weighted sum of all input positions
s = 22
x0 = 60
l4_w = [0.5, 0.3, 0.4, 1.2, 0.3, 0.3, 0.5, 3.2, 0.4]
g = row(x0, 50, 9, s, {7: VT}) + row(x0, 150, 9, s, {3: V})
for i, w in enumerate(l4_w):
    g += f'<line x1="{x0 + i * s + s / 2}" y1="{50 + s + 3}" x2="{x0 + 3 * s + s / 2}" y2="148" stroke="{V}" stroke-width="{0.6 + w:.1f}"/>'
g += (
    t(x0, 40, "(B, L, D)", "m")
    + t(272, 58, "ver weg,", "c")
    + t(272, 74, "toch relevant", "c")
    + t(272, 166, "(B, L, D)", "m")
    + t(200, 212, "gewicht naar relevantie, niet naar afstand", "c", "middle")
)
PICTURES["4.1"] = ("Wegen naar relevantie", g)

# 4.2 kernel, state and attention on the same axis; attention needs positions
s = 22
x0 = 120
g = row(x0, 34, 8, s, {3: VT, 4: VT, 5: VT})
g += f'<rect x="{x0 + 3 * s - 3}" y="31" width="{3 * s + 6}" height="{s + 6}" fill="none" stroke="{V}" stroke-width="2.5"/>'
g += row(x0, 94, 8, s, {4: VT}) + arrow(x0 + 11, 84, x0 + 4 * s + 11, 84, "v", 2.5)
g += "".join(
    f'<circle cx="{x0 + i * s + 11}" cy="84" r="3" fill="{V}"/>' for i in range(4)
)
g += row(x0, 160, 8, s, {4: VT}) + "".join(
    t(x0 + i * s + 11, 175, str(i + 1), "n", "middle") for i in range(8)
)
for i in range(8):
    if i != 4:
        a, b = x0 + 4 * s + 11, x0 + i * s + 11
        g += line(
            f"M {a} 160 Q {(a + b) / 2} {160 - 10 - 7 * abs(i - 4)} {b} 160", V, 1.5
        )
for y, name, sub in (
    (34, "kernel", "alleen buren"),
    (94, "state", "stap voor stap"),
    (160, "attention", "alles tegelijk"),
):
    g += t(14, y + 9, name, "l v") + t(14, y + 25, sub, "c")
g += t(304, 168, "+ positional", "c") + t(304, 184, "encoding", "c")
g += t(200, 212, "zonder positie kent attention geen volgorde", "c", "middle")
PICTURES["4.2"] = ("Drie manieren langs een as", g)

# 4.3 one step across any distance, and an L x L weight matrix as the price
g = t(94, 34, "winst", "l", "middle") + t(298, 34, "kosten", "l", "middle")
g += row(24, 110, 7, 20) + path("M 34 108 Q 94 40 154 108", "v", 2.5)
g += t(94, 66, "1 stap", "l v", "middle") + t(
    94, 152, "alle posities tegelijk", "c", "middle"
)
g += line("M 196 30 L 196 180", RULE, 1)
for gx, gy, n in ((222, 90, 4), (290, 50, 8)):
    g += "".join(
        cell(gx + c * 10, gy + r * 10, 10, VT, V, 1) for r in range(n) for c in range(n)
    )
g += t(242, 150, "L = 4", "m", "middle") + t(330, 150, "L = 8", "m", "middle")
g += t(242, 168, "16", "m v", "middle") + t(330, 168, "64", "m v", "middle")
g += t(200, 212, "dubbele lengte, vier keer zoveel gewichten", "c", "middle")
PICTURES["4.3"] = ("Overal één stap, kwadratisch", g)

# 4.4 three heads weigh the same tokens differently; the mask zeroes the padding
s = 26
x0 = 110
g = row(x0, 46, 5, s, {i: PT for i in range(5)})
g += "".join(
    cell(x0 + i * s, 46, s, SOFT, RULE, 1)
    + t(x0 + i * s + s / 2, 63, "pad", "n", "middle")
    for i in range(5, 8)
)
g += t(14, 63, "tokens", "m")
l4_heads = [
    [0.2, 0.6, 1.0, 0.3, 0.1],
    [0.9, 0.1, 0.2, 0.1, 0.6],
    [0.1, 0.3, 0.2, 1.0, 0.5],
]
for k, ws in enumerate(l4_heads):
    y = 88 + k * 32
    g += t(14, y + 17, f"head {k + 1}", "m")
    for i in range(8):
        x = x0 + i * s
        if i < 5:
            h = 22 * ws[i]
            g += cell(x, y, s, BG, RULE, 1)
            g += f'<rect x="{x + 4}" y="{y + 24 - h:.1f}" width="{s - 8}" height="{h:.1f}" fill="{VT}" stroke="{V}" stroke-width="1.5"/>'
        else:
            g += cell(x, y, s, SOFT, RULE, 1) + t(
                x + s / 2, y + 18, "0", "m v", "middle"
            )
g += f'<rect x="{x0 + 5 * s - 3}" y="43" width="{3 * s + 6}" height="138" fill="none" stroke="{V}" stroke-width="2" stroke-dasharray="5 4"/>'
g += t(x0 + 6.5 * s, 34, "mask", "l v", "middle")
g += t(200, 210, "padding krijgt gewicht nul, in elke head", "c", "middle")
PICTURES["4.4"] = ("Mask en heads", g)

# 4.5 the same sentence as letter, subword and word tokens under a 4-token budget
l4_text = "de rechter verwierp het"
l4_cw = 7.5
x0 = 100
g = t(x0, 38, "budget: 4 tokens", "l v")
for y, name, toks in (
    (54, "letters", list(l4_text)),
    (100, "subword", ["de", " recht", "er", " ver", "wierp", " het"]),
    (146, "woorden", ["de", " rechter", " verwierp", " het"]),
):
    pos = 0
    for k, tok in enumerate(toks):
        inside = k < 4
        g += f'<rect x="{x0 + pos * l4_cw}" y="{y}" width="{len(tok) * l4_cw}" height="22" fill="{VT if inside else BG}" stroke="{V if inside else RULE}" stroke-width="{1.5 if inside else 1}"/>'
        pos += len(tok)
    g += t(14, y + 15.5, name, "l") + t(x0, y + 15.5, l4_text, "m")
    g += t(286, y + 15.5, f"{len(toks)} tokens", "m")
g += t(200, 205, "hoeveel tekst past, hangt af van de tokenizer", "c", "middle")
PICTURES["4.5"] = ("Tokens zijn een budget", g)

# 4.6 a long document cut into overlapping chunks becomes a (chunks, dim) matrix
l4_starts = [14, 85, 156, 227, 298]
g = t(14, 34, "document, 2400 tokens", "m")
g += f'<rect x="14" y="44" width="372" height="14" fill="{PT}" stroke="{P}" stroke-width="1.5"/>'
g += "".join(
    f'<rect x="{x}" y="68" width="17" height="28" fill="{VT}"/>' for x in l4_starts[1:]
)
g += "".join(
    f'<rect x="{x}" y="{68 if k % 2 == 0 else 84}" width="88" height="12" fill="none" stroke="{V}" stroke-width="2"/>'
    for k, x in enumerate(l4_starts)
)
g += t(93.5, 114, "overlap", "c", "middle") + t(
    386, 114, "chunk = context window", "c", "end"
)
g += t(14, 155, "(5, 512)", "m") + arrow(80, 150, 108, 150)
g += box(112, 136, 92, 28, "encoder", "p", PT) + arrow(206, 150, 234, 150)
g += grid(240, 128, 5, 8, 9) + t(320, 155, "(5, dim)", "m")
g += t(200, 205, "kleinere chunks of meer overlap: meer rekentijd", "c", "middle")
PICTURES["4.6"] = ("Te lang: chunken", g)

# 4.7 padding rows at zero pull a plain mean towards the origin; a masked mean does not
g = t(24, 40, "kort document", "c")
for r in range(5):
    y = 50 + r * 16
    if r < 2:
        g += row(24, y, 6, 14, {i: PT for i in range(6)}) + t(
            14, y + 11, str(r + 1), "n"
        )
    else:
        g += "".join(cell(24 + i * 14, y, 14, SOFT, RULE, 1) for i in range(6))
        g += t(114, y + 11, "pad", "n")
g += line("M 190 178 L 380 178", RULE, 1) + line("M 210 30 L 210 190", RULE, 1)
g += line("M 210 178 L 320 85", MUT, 1, "4 3")
g += dot(210, 178, 4, SOFT, MUT) + t(216, 194, "pad = 0", "n")
g += dot(290, 60) + t(299, 57, "1", "n") + dot(350, 110) + t(359, 107, "2", "n")
g += dot(254, 140.8, 6, BG, V) + t(264, 146, "gewoon gemiddelde", "c")
g += dot(320, 85, 6, VT, V) + t(330, 92, "masked", "l v")
g += t(200, 212, "padding trekt het gemiddelde naar de oorsprong", "c", "middle")
PICTURES["4.7"] = ("Padding trekt naar nul", g)


# 4.8 two streams as circles: only the part the embedding lacks adds anything
def l4_crescent(xa: float, xb: float, y: float, r: float) -> str:
    xm = (xa + xb) / 2
    h = (r * r - ((xb - xa) / 2) ** 2) ** 0.5
    return f'<path d="M {xm} {y - h:.1f} A {r} {r} 0 1 1 {xm} {y + h:.1f} A {r} {r} 0 0 0 {xm} {y - h:.1f} Z" fill="{VT}"/>'


g = ""
for xa, xb in ((80, 110), (255, 335)):
    g += f'<circle cx="{xa}" cy="105" r="46" fill="{PT}"/>' + l4_crescent(
        xa, xb, 105, 46
    )
    g += "".join(
        f'<circle cx="{x}" cy="105" r="46" fill="none" stroke="{P}" stroke-width="1.5"/>'
        for x in (xa, xb)
    )
g += t(56, 48, "embedding", "c", "middle") + t(140, 48, "feature", "c", "middle")
g += t(250, 48, "embedding", "c", "middle") + t(340, 48, "feature", "c", "middle")
g += t(345, 110, "nieuw", "l v", "middle") + line("M 190 40 L 190 160", RULE, 1)
g += t(95, 180, "overlap: geen winst", "l", "middle") + t(
    295, 180, "complementair: winst", "l v", "middle"
)
g += t(200, 210, "meer parameters is geen reden", "c", "middle")
PICTURES["4.8"] = ("Fusie vraagt nieuwe informatie", g)

# 4.9 the pipeline from document to one vector, with the shape after every step
g = ""
l4_steps = [
    ("document", "g", "(T,)"),
    ("chunk", "v", "(chunks, L)"),
    ("embed via cache", "v", "(chunks, D)"),
    ("pad", "v", "(B, C, D) + mask"),
    ("masked mean", "v", "(B, D)"),
]
for k, (name, c, shape) in enumerate(l4_steps):
    y = 22 + 38 * k
    g += box(24, y, 140, 22, name, c, VT if c == "v" else BG)
    g += arrow(168, y + 11, 196, y + 11) + t(202, y + 15.5, shape, "m")
    if k < 4:
        g += arrow(94, y + 23, 94, y + 37)
g += t(200, 214, "schrijf de shape op vóór elke stap", "c", "middle")
PICTURES["4.9"] = ("Van document naar vector", g)

# 4.10 an ablation table: drop one stream per row, read the score against the spread
g = t(130, 40, "embedding", "c", "middle") + t(190, 40, "feature", "c", "middle")
g += t(310, 40, "score", "l", "middle")
g += line("M 240 166 L 380 166", RULE, 1)
for k, (name, on, sc) in enumerate(
    (
        ("fusie", (True, True), 352),
        ("− feature", (True, False), 320),
        ("− embedding", (False, True), 260),
    )
):
    y = 70 + 40 * k
    g += t(14, y + 5, name, "l")
    for j, o in enumerate(on):
        x = 119 + 60 * j
        g += (
            cell(x, y - 11, 22, PT, P)
            if o
            else f'<rect x="{x}" y="{y - 11}" width="22" height="22" fill="{BG}" stroke="{V}" stroke-width="2" stroke-dasharray="3 3"/>'
        )
    g += line(
        f"M {sc - 9} {y} L {sc + 9} {y} M {sc - 9} {y - 5} L {sc - 9} {y + 5} M {sc + 9} {y - 5} L {sc + 9} {y + 5}",
        P,
    ) + dot(sc, y, 5, VT if k == 0 else PT, V if k == 0 else P)
g += t(310, 182, "spreiding tussen runs", "c", "middle")
g += t(200, 208, "één variabele tegelijk, de rest blijft gelijk", "c", "middle")
PICTURES["4.10"] = ("De ablatie beslist", g)

# 4.11 the search loops over train and valid; test is touched once at the end
g = t(20, 32, "search", "l v")
g += box(20, 40, 240, 30, "chunk × padding × aggregatie", "v", VT, "c")
g += path("M 120 72 L 120 146") + path("M 250 148 L 250 74")
g += t(185, 114, "24 trials", "c", "middle")
for x, w, name in ((20, 210, "train"), (230, 70, "valid"), (300, 80, "test")):
    g += box(x, 150, w, 26, name, "p", PT if name != "test" else BG)
g += path("M 262 55 C 340 55, 340 90, 340 146", "v", 2)
g += t(352, 106, "1×", "big v")
g += t(200, 208, "rapporteer op de split die de search nooit zag", "c", "middle")
PICTURES["4.11"] = ("Test pas aan het eind", g)

# 4.12 the class design puts attention over chunks; the class publishes and builds on it
g = t(14, 32, "ontwerp", "c")
g += box(14, 42, 72, 28, "uitspraak", "g", BG, "c") + arrow(88, 56, 106, 56)
g += row(110, 45, 4, 22, {i: PT for i in range(4)}) + arrow(200, 56, 220, 56)
g += box(224, 42, 80, 28, "attention", "v", VT) + arrow(306, 56, 326, 56)
g += box(330, 42, 56, 28, "head", "p")
g += t(264, 88, "over chunks", "c", "middle")
g += t(14, 108, "als klas", "c")
g += box(14, 116, 110, 28, "publiceren", "p") + arrow(126, 130, 150, 130)
g += box(154, 116, 100, 28, "bekijken", "p") + arrow(256, 130, 276, 130)
g += box(280, 116, 106, 28, "voortbouwen", "v", VT)
g += path("M 333 146 C 333 192, 69 192, 69 146", "v", 2.5)
g += t(200, 204, "met bronvermelding", "l v", "middle")
PICTURES["4.12"] = ("SCOTUS als klas", g)


def l5_square(x: float, y: float, fill: str = PT, stroke: str = P, r: float = 6) -> str:
    return f'<rect x="{x - r}" y="{y - r}" width="{2 * r}" height="{2 * r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'


def l5_panel(x: float, y: float, w: float, h: float) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{BG}" stroke="{RULE}"/>'
    )


# 5.1 hidden cells of the input become the target of the prediction
g = t(24, 40, "invoer, deels verborgen", "c") + row(24, 50, 8, 22, {5: SOFT, 6: SOFT})
g += t(145, 66, "?", "m", "middle") + t(167, 66, "?", "m", "middle")
g += arrow(204, 61, 232, 61) + box(236, 44, 84, 34, "encoder", "p", PT)
g += arrow(322, 61, 338, 61) + row(342, 50, 2, 22, {0: PT, 1: PT})
g += t(386, 40, "voorspeld", "c", "end")
g += row(24, 140, 8, 22, {5: VT, 6: VT}) + t(24, 182, "dezelfde data, compleet", "c")
g += path("M 156 140 C 156 105, 364 120, 364 76", "v", 2.5)
g += t(276, 134, "het doel", "l v")
g += t(200, 210, "geen labels: de data levert het antwoord", "c", "middle")
PICTURES["5.1"] = ("De data levert het antwoord", g)

# 5.2 the same items ordered by two pretext tasks: by colour or by shape
g = l5_panel(20, 28, 170, 142) + l5_panel(210, 28, 170, 142)
g += t(105, 48, "reconstructie", "l", "middle") + t(
    295, 48, "contrastief", "l", "middle"
)
g += t(295, 66, "augmentatie: kleur", "c", "middle")
for x, y, sq, dark in (
    (50, 92, False, True),
    (72, 86, True, True),
    (62, 110, True, True),
    (84, 106, False, True),
    (128, 104, False, False),
    (150, 98, True, False),
    (140, 126, True, False),
    (162, 122, False, False),
    (238, 96, False, True),
    (260, 90, False, False),
    (248, 116, False, False),
    (270, 112, False, True),
    (318, 108, True, False),
    (340, 102, True, True),
    (328, 130, True, True),
    (350, 126, True, False),
):
    fill = P if dark else BG
    g += l5_square(x, y, fill) if sq else dot(x, y, 6, fill)
g += t(105, 160, "vergeet de vorm", "l v", "middle") + t(
    295, 160, "vergeet de kleur", "l v", "middle"
)
g += t(200, 208, "de pretext task kiest wat de ruimte vergeet", "c", "middle")
PICTURES["5.2"] = ("Kiezen wat de ruimte vergeet", g)

# 5.3 a funnel from input to a narrow latent and back, the loss closing on the input
g = "".join(cell(80, 40 + i * 16, 16, PT) for i in range(8))
g += f'<polygon points="100,40 190,88 190,120 100,168" fill="{BG}" stroke="{P}" stroke-width="2"/>'
g += "".join(cell(198, 88 + i * 16, 16, VT, V) for i in range(2))
g += f'<polygon points="222,88 312,40 312,168 222,120" fill="{BG}" stroke="{P}" stroke-width="2"/>'
g += "".join(cell(316, 40 + i * 16, 16, PT) for i in range(8))
g += t(145, 108, "encoder", "l p", "middle") + t(267, 108, "decoder", "l p", "middle")
g += t(88, 32, "invoer", "c", "middle") + t(324, 32, "uitvoer", "c", "middle")
g += t(206, 80, "latent", "c", "middle")
g += path("M 324 172 C 324 202, 88 202, 88 174", "v", 2.5)
g += t(206, 188, "loss", "l v", "middle")
g += t(200, 216, "het doel is de invoer zelf", "c", "middle")
PICTURES["5.3"] = ("Smal midden, invoer als doel", g)

# 5.4 one item, two augmentations, one shared encoder, two vectors pulled together
g = t(39, 86, "item", "c", "middle") + grid(18, 94, 3, 3, 14)
g += arrow(62, 106, 92, 74) + arrow(62, 124, 92, 156)
g += box(96, 54, 52, 30, "aug 1", "v", VT) + box(96, 146, 52, 30, "aug 2", "v", VT)
g += arrow(150, 69, 168, 69) + arrow(150, 161, 168, 161)
g += "".join(
    row(172, 51 + r * 12, 3, 12, {(r + 1) % 3: PT}) for r in range(3)
) + "".join(row(172, 143 + r * 12, 3, 12, {r: PT}) for r in range(3))
g += arrow(210, 69, 232, 69) + arrow(210, 161, 232, 161)
g += box(236, 50, 68, 130, "encoder", "p", PT) + t(270, 138, "gedeeld", "c", "middle")
g += arrow(306, 69, 326, 69) + arrow(306, 161, 326, 161)
g += dot(334, 69, 6) + dot(334, 161, 6)
g += line("M 334 80 L 334 104 M 334 126 L 334 150", V, 2)
g += t(386, 120, "samen", "l v", "end")
g += t(200, 208, "wat de augmentatie verandert, vergeet de ruimte", "c", "middle")
PICTURES["5.4"] = ("Twee views, één encoder", g)

# 5.5 three moments of training: the loss falls while the vectors shrink to one point
g = ""
for i, (ep, scale, lo, sp) in enumerate(
    (
        ("epoch 1", 1.0, "2.3", "0.9"),
        ("epoch 10", 0.4, "1.1", "0.4"),
        ("epoch 50", 0.0, "0.1", "0.0"),
    )
):
    cx = 75 + i * 125
    g += l5_panel(cx - 55, 28, 110, 108) + t(cx, 46, ep, "c", "middle")
    for k, (dx, dy) in enumerate(
        ((-35, -22), (-10, -30), (25, -18), (35, 10), (10, 30), (-25, 25), (0, 0))
    ):
        if scale:
            g += dot(cx + dx * scale, 92 + dy * scale, 5)
        else:
            g += dot(cx + k * 0.6, 92, 6, VT, V)
    g += t(cx, 156, f"loss {lo}", "m p", "middle")
    g += t(cx, 176, f"spreiding {sp}", "m v", "middle")
g += t(200, 208, "meet de spreiding, niet de loss", "c", "middle")
PICTURES["5.5"] = ("Collapse: alles op één punt", g)

# 5.6 a tidy-looking space the probe cannot split next to a messy one it can
g = l5_panel(20, 26, 170, 124) + l5_panel(210, 26, 170, 124)
for x, y in ((64, 78), (78, 92), (134, 100), (148, 114)):
    g += dot(x, y)
for x, y in ((76, 76), (62, 94), (146, 98), (132, 114)):
    g += l5_square(x, y)
g += line("M 88 140 L 122 36", V, 2, "5 4")
for x, y in ((240, 90), (260, 125), (290, 130), (230, 135), (322, 140)):
    g += dot(x, y)
for x, y in ((280, 50), (320, 70), (350, 95), (300, 42), (362, 70)):
    g += l5_square(x, y)
g += line("M 220 40 L 370 140", V, 2)
g += t(105, 168, "strak geclusterd", "l", "middle") + t(
    295, 168, "verspreid", "l", "middle"
)
g += t(105, 188, "probe 50 %", "m v", "middle") + t(
    295, 188, "probe 92 %", "m v", "middle"
)
g += t(200, 214, "lees de ruimte af aan de probe", "c", "middle")
PICTURES["5.6"] = ("De probe is de meetlat", g)

# 5.7 two pipelines: the autoencoder loss sits on pixels, the JEPA loss on latent vectors
g = t(20, 30, "autoencoder", "l") + grid(20, 40, 3, 3, 12)
g += arrow(58, 58, 72, 58) + box(76, 42, 78, 32, "encoder", "p", PT)
g += arrow(156, 58, 170, 58) + box(174, 42, 78, 32, "decoder", "p")
g += arrow(254, 58, 268, 58) + grid(272, 40, 3, 3, 12)
g += t(330, 54, "loss op", "l v") + t(330, 70, "pixels", "l v")
g += t(20, 118, "JEPA", "l")
g += "".join(
    row(20, 128 + r * 12, 3, 12, {c: VT for c in (1, 2)} if r > 0 else {})
    for r in range(3)
)
g += arrow(58, 146, 72, 146) + box(76, 130, 78, 32, "encoder", "p", PT)
g += arrow(156, 146, 170, 146) + box(174, 130, 78, 32, "predictor", "p")
g += arrow(254, 146, 268, 146) + row(272, 140, 4, 12, {i: PT for i in range(4)})
g += t(330, 142, "loss op", "l v") + t(330, 158, "latent", "l v")
g += t(20, 182, "gemaskeerd", "c")
g += t(200, 210, "voorspel wat er hoort, niet hoe het eruitziet", "c", "middle")
PICTURES["5.7"] = ("Voorspellen in latent space", g)

# 5.8 one patch grid: pooled it keeps what, with the patch axis it also keeps where
g = t(68, 56, "patch tokens", "c", "middle")
g += "".join(row(24, 64 + r * 22, 4, 22, {3: P} if r == 0 else {}) for r in range(4))
g += arrow(116, 74, 170, 56) + box(174, 40, 70, 28, "pooling", "v", VT)
g += arrow(246, 54, 270, 54) + row(274, 43, 4, 22, {i: PT for i in range(4)})
g += t(318, 34, "(B, D)", "n", "middle") + t(386, 88, "weet wat, niet waar", "c", "end")
g += arrow(116, 140, 266, 140) + t(190, 132, "patch-as blijft", "c", "middle")
g += "".join(row(274, 114 + r * 13, 4, 13, {3: P} if r == 0 else {}) for r in range(4))
g += t(300, 108, "(B, P, D)", "n", "middle") + t(386, 184, "weet ook waar", "c", "end")
g += t(200, 212, "pooling middelt de plaats weg", "c", "middle")
PICTURES["5.8"] = ("Pooling gooit de plaats weg", g)

# 5.9 a production line: unlabelled data and a pretext task make the encoder lesson 1 caches
g = t(108, 52, "deze les", "l v", "middle") + t(302, 52, "les 1", "l p", "middle")
g += line("M 212 38 L 212 140", RULE, 1, "4 4")
g += box(18, 70, 60, 40, "data", "g") + t(48, 130, "ongelabeld", "c", "middle")
g += arrow(80, 90, 96, 90) + box(100, 70, 100, 40, "pretext task", "v", VT)
g += arrow(202, 90, 222, 90) + box(226, 70, 74, 40, "encoder", "p", PT)
g += arrow(302, 90, 318, 90) + box(322, 70, 60, 40, "cache", "p")
g += t(263, 130, "halffabrikaat", "l p", "middle")
g += t(200, 205, "de bevroren encoder is zo'n training", "c", "middle")
PICTURES["5.9"] = ("Een halffabrikaat", g)

# 5.10 one dataset through three encoders into the same probe, three scores out
g = box(16, 95, 52, 34, "data", "g")
for y, lab, w in (
    (50, "autoencoder", 40),
    (112, "contrastief", 52),
    (174, "supervised", 64),
):
    g += path(f"M 70 112 L 92 {y}") + box(96, y - 15, 110, 30, lab, "p", PT)
    g += arrow(208, y, 228, y) + arrow(298, y, 314, y)
    g += f'<rect x="318" y="{y - 7}" width="{w}" height="14" fill="{PT}" stroke="{P}" stroke-width="1.5"/>'
g += box(232, 35, 64, 154, "probe", "v", VT) + t(264, 150, "dezelfde", "c", "middle")
g += t(200, 214, "één meetlat, drie ruimtes", "c", "middle")
PICTURES["5.10"] = ("Drie ruimtes, één probe", g)

# 5.11 a table of trials: the lowest loss is the collapsed one, the probe picks another
g = f'<rect x="18" y="84" width="284" height="20" fill="{VT}"/>'
g += f'<rect x="256" y="110" width="44" height="20" fill="none" stroke="{P}" stroke-width="2"/>'
g += t(24, 44, "trial", "m") + t(115, 44, "loss", "m", "middle")
g += t(203, 44, "spreiding", "m v", "middle") + t(278, 44, "probe", "m", "middle")
g += line("M 18 54 L 302 54", RULE, 1)
for i, (lo, sp, pr) in enumerate(
    (
        ("0.82", "0.61", "71 %"),
        ("0.05", "0.00", "12 %"),
        ("0.64", "0.48", "84 %"),
        ("0.71", "0.55", "78 %"),
    )
):
    y = 72 + i * 26
    g += t(24, y, f"trial {i + 1}", "m") + t(115, y, lo, "m", "middle")
    g += t(203, y, sp, "m v" if i == 1 else "m", "middle") + t(
        278, y, pr, "m", "middle"
    )
g += t(312, 98, "collapse", "c v") + t(312, 124, "gekozen", "l p")
g += t(200, 200, "zoek op de probe, met de spreiding als alarm", "c", "middle")
PICTURES["5.11"] = ("Zoeken op de meting", g)

# 5.12 object left or right: the pooled vectors are equal, the patch rows differ
g = t(20, 36, "beeld", "c") + t(205, 36, "gepoold", "l", "middle")
g += t(325, 36, "patch-as", "l", "middle")
for y, pos in ((56, 0), (120, 3)):
    g += row(20, y, 4, 20, {pos: P}) + row(185, y, 2, 20, {0: PT, 1: PT})
    g += row(285, y, 4, 20, {pos: P})
g += t(205, 106, "=", "big v", "middle") + t(325, 106, "≠", "big p", "middle")
g += t(20, 106, "links of rechts", "c")
g += t(205, 168, "kan het niet zien", "l v", "middle") + t(
    325, 168, "kan het zien", "l p", "middle"
)
g += t(200, 208, "de taak vraagt waar: kies de patch-as", "c", "middle")
PICTURES["5.12"] = ("Plaats telt: houd de patch-as", g)


def l6_rect(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str = BG,
    stroke: str = P,
    sw: float = 1.5,
) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def l6_bell(cx: float, base: float, sigma: float, h: float, span: float = 3) -> str:
    pts = []
    for i in range(41):
        z = -span + 2 * span * i / 40
        pts.append(f"{cx + z * sigma:.1f} {base - h * 2.718281828 ** (-z * z / 2):.1f}")
    return "M " + " L ".join(pts)


# 6.1 split in time with train-only statistics, a window inside each split, seasonal naive
s = 18
g = (
    row(30, 42, 14, s, {i: PT for i in range(5)})
    + t(75, 34, "train", "l p", "middle")
    + t(156, 34, "valid", "c", "middle")
    + t(237, 34, "test", "c", "middle")
)
g += line("M 30 66 L 30 70 L 120 70 L 120 66", V, 1.5) + t(
    75, 86, "μ, σ uit train", "c v", "middle"
)
g += row(30, 104, 14, s, {0: PT, 1: PT, 2: PT, 3: PT, 4: VT, 8: VT, 13: VT})
for x0 in (30 + 2 * s, 30 + 6 * s, 30 + 11 * s):
    g += f'<rect x="{x0 - 2}" y="102" width="{3 * s + 4}" height="{s + 4}" fill="none" stroke="{V}" stroke-width="2"/>'
g += line("M 120 36 L 120 128 M 192 36 L 192 128", V, 2.5)
g += t(296, 56, "1 split", "l") + t(296, 72, "in de tijd", "c")
g += t(296, 118, "2 window", "l") + t(296, 134, "per split", "c")
g += line("M 30 176 q 15 -28 30 0 t 30 0 t 30 0 t 30 0 t 30 0 t 30 0", P)
g += line("M 210 176 q 15 -28 30 0 t 30 0", V, 2, "5 3")
g += path("M 166 156 Q 210 136 252 156", "v")
g += t(296, 180, "3 baseline", "l")
g += t(150, 206, "seasonal naive: herhaal de vorige dag", "c", "middle")
PICTURES["6.1"] = ("Eerst splitsen, dan windowen", g)

# 6.2 point forecast against a fan that crosses a threshold
hist = "q 10 -20 20 0 t 20 0 t 20 0 t 20 0"
g = t(105, 34, "puntvoorspelling", "l", "middle") + t(
    295, 34, "verdeling", "l v", "middle"
)
for x0 in (30, 220):
    g += line(f"M {x0} 70 L {x0 + 150} 70", MUT, 1, "4 3") + t(x0, 62, "grens", "n")
    g += line(f"M {x0} 120 {hist}", P)
g += line("M 110 120 L 180 112", V, 2.5)
g += f'<path d="M 300 120 L 370 50 L 370 158 Z" fill="{VT}" stroke="none"/>'
g += f'<path d="M 350 70 L 370 50 L 370 70 Z" fill="{V}" stroke="none"/>'
g += line("M 300 120 L 370 110", V, 2.5)
g += line("M 200 26 L 200 182", RULE, 1)
g += t(105, 182, "zegt niets over de grens", "c", "middle") + t(
    295, 182, "kans boven de grens: 8 %", "c v", "middle"
)
g += t(200, 210, "vraag over een grens: de breedte is het antwoord", "c", "middle")
PICTURES["6.2"] = ("De breedte is het antwoord", g)

# 6.3 stacked uncertainty against data: the epistemic layer shrinks, the aleatoric floor stays
g = line("M 50 30 L 50 160 L 370 160", MUT, 1) + t(56, 28, "onzekerheid", "c")
g += f'<path d="M 60 110 L 360 110 L 360 160 L 60 160 Z" fill="{PT}" stroke="none"/>'
g += f'<path d="M 60 40 C 140 90, 220 106, 360 110 L 60 110 Z" fill="{VT}" stroke="none"/>'
g += line("M 60 40 C 140 90, 220 106, 360 110", V, 2) + line("M 60 110 L 360 110", P, 2)
g += t(66, 100, "epistemisch", "l v")
g += t(210, 140, "aleatorisch: de wereld varieert", "l p", "middle")
g += t(210, 178, "meer data →", "c", "middle")
g += t(200, 208, "meer data haalt alleen de epistemische laag weg", "c", "middle")
PICTURES["6.3"] = ("Meer data, minder epistemisch", g)

# 6.4 two forecasts with the same middle: equal MAE, different CRPS
g = line("M 130 34 L 130 170", RULE, 1, "4 3") + t(130, 28, "midden", "n", "middle")
for base, sig, h, lab, crps in (
    (80, 10, 44, "scherp", "0.30"),
    (160, 30, 14, "breed", "0.42"),
):
    g += f'<path d="{l6_bell(130, base, sig, h)} Z" fill="{PT}" stroke="{P}" stroke-width="1.5"/>'
    g += line(f"M 30 {base} L 230 {base}", MUT, 1)
    g += line(f"M 140 {base - 7} L 140 {base + 7}", INK, 2) + t(
        144, base + 18, "waarheid", "n"
    )
    g += t(30, base - 22, lab, "c")
    g += t(262, base - 18, "MAE  0.5", "m") + t(262, base, f"CRPS {crps}", "m v")
g += t(200, 208, "zelfde midden, zelfde MAE; CRPS ziet de breedte", "c", "middle")
PICTURES["6.4"] = ("CRPS ziet de breedte", g)

# 6.5 a Gaussian head forces a bell on skewed data, a quantile head reads the quantiles
bars = [6, 22, 34, 30, 22, 15, 10, 7, 4, 2]
g = t(20, 46, "Gaussische head", "l v") + t(20, 66, "(B, 2)  μ, σ", "m")
g += t(20, 84, "loss: NLL", "m") + t(20, 102, "neemt een bel aan", "c")
g += t(20, 140, "quantile head", "l v") + t(20, 160, "(B, 3)  q10 q50 q90", "m")
g += t(20, 178, "loss: pinball", "m") + t(20, 196, "geen vorm nodig", "c")
g += line("M 20 118 L 380 118", RULE, 1)
for base in (104, 196):
    g += "".join(
        l6_rect(210 + i * 16, base - b * 1.6, 16, b * 1.6, PT, P, 1)
        for i, b in enumerate(bars)
    )
g += line(l6_bell(271, 104, 32, 48, 2.6), V, 2.5)
for x, q in ((233, "q10"), (266, "q50"), (319, "q90")):
    g += line(f"M {x} 196 L {x} 140", V, 2.5) + t(x, 134, q, "n v", "middle")
PICTURES["6.5"] = ("Quantiles nemen geen vorm aan", g)

# 6.6 calibration plot below the diagonal, and narrow intervals that miss the truth
g = line("M 44 36 L 44 166 L 174 166", MUT, 1) + line(
    "M 44 166 L 174 36", MUT, 1, "4 3"
)
g += t(44, 30, "gehaald", "c") + t(174, 30, "ideaal", "n", "end")
g += line("M 161 166 L 161 88 M 44 88 L 161 88", RULE, 1, "3 3")
g += line("M 44 166 Q 120 150 161 88", V, 2) + dot(161, 88, 5, VT, V)
g += t(161, 180, "90 %", "n", "middle") + t(40, 92, "60 %", "n", "end")
g += t(96, 180, "gevraagd", "c", "middle")
g += line("M 205 26 L 205 182", RULE, 1)
g += t(300, 40, "90 %-intervallen", "l", "middle")
cs = [96, 104, 92, 100, 108, 98, 94, 102, 100, 96]
for i, c in enumerate(cs):
    x = 233 + i * 15
    g += line(
        f"M {x} {c - 10} L {x} {c + 10} M {x - 4} {c - 10} L {x + 4} {c - 10} M {x - 4} {c + 10} L {x + 4} {c + 10}",
        P,
        1.5,
    )
    if i in (1, 4, 6, 8):
        g += dot(x, c + (18 if i % 4 == 0 else -18), 4, VT, V)
    else:
        g += dot(x, c + (4 if i % 2 else -4), 3, PT, P)
g += t(300, 150, "smal", "l p", "middle") + t(
    300, 168, "dekt 6 van 10", "c v", "middle"
)
g += t(200, 208, "scherp, en toch slecht gekalibreerd", "c", "middle")
PICTURES["6.6"] = ("Scherp is niet gekalibreerd", g)

# 6.7 split conformal: a quantile of held-out scores sets a threshold, the set size varies
scores = [
    0.02,
    0.05,
    0.08,
    0.1,
    0.13,
    0.16,
    0.19,
    0.23,
    0.27,
    0.31,
    0.37,
    0.44,
    0.52,
    0.6,
    0.68,
    0.76,
    0.83,
    0.88,
    0.95,
    0.98,
]
g = t(20, 64, "kalibratieset", "l p") + t(20, 82, "apart gehouden", "c")
g += line("M 140 80 L 360 80", MUT, 1)
g += "".join(dot(140 + sc * 220, 58 if i % 2 else 68, 4) for i, sc in enumerate(scores))
g += line("M 342 44 L 342 84", V, 2.5) + t(342, 38, "q̂", "l v", "middle")
g += t(334, 38, "90 % eronder", "c v", "end")
g += t(250, 96, "score: 1 − p(juiste klasse)", "n", "middle")
g += t(140, 118, "set = klassen met score onder q̂", "n")
for y, lab, members, size in (
    (124, "duidelijk", (3,), "1 klasse"),
    (152, "twijfelachtig", (1, 2, 3, 5, 7), "5 klassen"),
):
    g += t(20, y + 12, lab, "l") + row(140, y, 10, 16, {k: VT for k in members})
    g += t(310, y + 12, size, "m v")
g += t(200, 200, "dekt 90 %, zolang test lijkt op kalibratie", "c", "middle")
PICTURES["6.7"] = ("Dekking zonder aanname", g)

# 6.8 expected cost against threshold: the minimum lies far from 0.5
g = line("M 50 30 L 50 150 L 350 150", MUT, 1) + t(54, 28, "verwachte kosten", "c")
g += t(370, 28, "C_fn = 5 × C_fp", "m", "end")
g += line("M 56 108 C 70 128, 86 136, 101 136 C 146 136, 250 80, 344 46", P, 2)
g += line("M 200 105 L 200 150", MUT, 1, "4 3") + dot(200, 105, 5, BG, MUT)
g += line("M 101 136 L 101 150", V, 1.5) + dot(101, 136, 5, VT, V)
g += t(200, 164, "0.5", "n", "middle") + t(200, 182, "gewoonte", "c", "middle")
g += t(101, 164, "0.17", "n v", "middle") + t(101, 182, "beste", "l v", "middle")
g += t(350, 164, "drempel →", "c", "end")
g += t(200, 210, "lager dan 0.5: meer recall, minder precision", "c", "middle")
PICTURES["6.8"] = ("Drempel is een beslissing", g)

# 6.9 two thresholds make three outcomes: no, refer, yes
ps = [0.05, 0.12, 0.2, 0.3, 0.45, 0.5, 0.58, 0.8, 0.9, 0.95]
g = t(30, 60, "0", "n", "middle") + t(370, 60, "1", "n", "middle")
g += t(200, 60, "p(positief)", "n", "middle")
g += (
    l6_rect(30, 90, 120, 20, PT, P)
    + l6_rect(150, 90, 100, 20, VT, V)
    + l6_rect(250, 90, 120, 20, PT, P)
)
g += line("M 150 84 L 150 116 M 250 84 L 250 116", V, 2.5)
g += "".join(
    dot(30 + 340 * p, 78, 4, VT if 0.36 < p < 0.64 else PT, V if 0.36 < p < 0.64 else P)
    for p in ps
)
g += (
    arrow(90, 114, 82, 136) + arrow(200, 114, 200, 136, "v") + arrow(310, 114, 318, 136)
)
g += box(42, 140, 80, 32, "nee", "p") + box(140, 140, 120, 32, "doorverwijzen", "v", VT)
g += box(278, 140, 80, 32, "ja", "p")
g += t(200, 206, "weten wat je niet weet is ook een antwoord", "c", "middle")
PICTURES["6.9"] = ("Niet antwoorden mag", g)

# 6.10 one trunk, a Gaussian and a quantile head, both scored in CRPS against seasonal naive
g = t(14, 108, "(B, W)", "m") + arrow(62, 104, 78, 104)
g += box(80, 89, 60, 30, "trunk", "p", PT)
g += path("M 142 98 L 164 66") + path("M 142 110 L 164 144")
g += box(166, 50, 76, 28, "Gauss", "v", VT) + t(204, 44, "(B, H, 2)", "n v", "middle")
g += box(166, 132, 76, 28, "quantile", "v", VT) + t(
    204, 176, "(B, H, Q)", "n v", "middle"
)
g += t(262, 68, "CRPS", "l v")
for y, lab, w, c in (
    (92, "naïef", 62, "p"),
    (116, "Gauss", 46, "v"),
    (140, "quantile", 41, "v"),
):
    fill, stroke = (PT, P) if c == "p" else (VT, V)
    g += t(262, y + 4, lab, "n") + l6_rect(318, y - 7, w, 14, fill, stroke)
g += line("M 380 80 L 380 152", MUT, 1, "4 3")
g += t(200, 208, "beide heads tegen dezelfde naïeve baseline", "c", "middle")
PICTURES["6.10"] = ("Beide heads tegen de baseline", g)

# 6.11 fit q-hat on valid, measure coverage on test, read it off the calibration plot
s = 18
g = row(30, 44, 14, s, {i: PT for i in range(5)})
g += line("M 120 38 L 120 68 M 192 38 L 192 68", MUT, 1.5)
g += path("M 160 40 Q 196 14 233 40", "v")
g += t(75, 80, "train", "c p", "middle") + t(156, 80, "valid", "c p", "middle")
g += t(237, 80, "test", "c p", "middle")
g += t(156, 98, "q̂ fitten", "c v", "middle") + t(
    237, 98, "dekking meten", "c v", "middle"
)
g += line("M 60 116 L 60 200 L 200 200", MUT, 1) + line(
    "M 60 200 L 200 116", MUT, 1, "4 3"
)
g += t(60, 110, "gehaald", "n") + t(130, 214, "gevraagd", "n", "middle")
for n, o in ((0.8, 0.81), (0.9, 0.9), (0.95, 0.94), (0.99, 0.99)):
    g += dot(60 + (n - 0.5) / 0.5 * 140, 200 - (o - 0.5) / 0.5 * 84, 4, VT, V)
g += t(220, 150, "op de diagonaal:", "c") + t(220, 168, "dekking gehaald", "l v")
PICTURES["6.11"] = ("Fit op valid, meet op test", g)

# 6.12 a cost matrix with refer as third action sets two thresholds on p
cols = ("alarm", "verwijs", "niets")
costs = (("ziek", ("0", "2", "10")), ("gezond", ("5", "2", "0")))
g = "".join(t(106 + j * 52, 36, c, "c", "middle") for j, c in enumerate(cols))
for i, (lab, vals) in enumerate(costs):
    y = 44 + i * 28
    g += t(20, y + 19, lab, "l")
    for j, v in enumerate(vals):
        bad = (i, j) in ((0, 2), (1, 0))
        g += l6_rect(
            80 + j * 52,
            y,
            52,
            28,
            VT if bad else BG,
            V if bad else RULE,
            1.5 if bad else 1,
        )
        g += t(106 + j * 52, y + 19, v, "m v" if bad else "m", "middle")
g += t(246, 63, "gemiste patiënt: 10", "c v") + t(246, 91, "vals alarm: 5", "c v")
g += arrow(158, 104, 158, 124)
g += (
    l6_rect(40, 134, 64, 18, PT, P)
    + l6_rect(104, 134, 128, 18, VT, V)
    + l6_rect(232, 134, 128, 18, PT, P)
)
g += line("M 104 128 L 104 158 M 232 128 L 232 158", V, 2.5)
g += t(40, 128, "0", "n", "middle") + t(360, 128, "1", "n", "middle")
g += t(104, 122, "0.2", "n v", "middle") + t(232, 122, "0.6", "n v", "middle")
g += t(300, 128, "p(ziek)", "n", "middle")
g += t(72, 172, "niets", "c", "middle") + t(168, 172, "verwijs", "c v", "middle")
g += t(296, 172, "alarm", "c", "middle")
g += t(200, 204, "de kosten kiezen de drempels, niet het model", "c", "middle")
PICTURES["6.12"] = ("Kosten kiezen de drempels", g)


def l7_rect(
    x: float, y: float, w: float, h: float, fill: str, stroke: str, sw: float = 1.5
) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


# 7.1 instruction and examples feed a frozen generalist; training is crossed out
g = (
    box(20, 64, 100, 28, "instructie", "v", VT)
    + box(20, 104, 100, 28, "voorbeelden", "v", VT)
    + path("M 122 78 L 156 96", "v")
    + path("M 122 118 L 156 104", "v")
    + box(160, 72, 120, 56, "generalist", "p", PT)
    + t(220, 146, "gewichten bevroren", "m", "middle")
    + arrow(282, 100, 312, 100)
    + t(318, 104, "antwoord", "l")
)
g += (
    t(220, 34, "training", "c", "middle")
    + arrow(220, 40, 220, 68, "g", 1.5, "4 3")
    + line("M 212 46 L 228 62 M 228 46 L 212 62", MUT, 2.5)
)
g += t(200, 205, "de hefboom zit in wat je meegeeft", "c", "middle")
PICTURES["7.1"] = ("Voeden, niet trainen", g)

# 7.2 head and generalist as two points on a score-by-cost plane
g = line("M 60 30 L 60 170 L 376 170", MUT, 1) + t(20, 40, "score", "c")
g += t(376, 160, "kosten", "c", "end")
g += line("M 60 110 L 90 110 M 95 115 L 95 170", RULE, 1, "4 3") + line(
    "M 60 70 L 315 70 M 320 75 L 320 170", RULE, 1, "4 3"
)
g += dot(95, 110, 6) + t(95, 96, "head", "l p", "middle")
g += dot(320, 70, 6, VT, V) + t(320, 56, "generalist", "l v", "middle")
g += t(95, 188, "bijna gratis", "m", "middle") + t(
    320, 188, "duur per item", "m", "middle"
)
g += t(200, 212, "één getal is een halve vergelijking", "c", "middle")
PICTURES["7.2"] = ("Score én kosten", g)

# 7.3 three outputs, each with its own verdict: reference, unreadable count, judged judge
g = ""
rows73 = [
    (30, "uitvoer", "m", "vs. goed antwoord", "accuracy"),
    (90, "?#!%", "m", "telt als uitkomst", "% onleesbaar"),
    (150, "judge", "m", "vs. jouw labels", "judge-accuracy"),
]
for y, left, cls, mid, right in rows73:
    g += (
        box(14, y, 80, 28, left, "p", PT, cls)
        + arrow(96, y + 14, 118, y + 14)
        + box(122, y, 140, 28, mid, "v", VT)
        + arrow(264, y + 14, 286, y + 14)
        + t(292, y + 19, right, "c")
    )
g += t(200, 210, "ook de beoordelaar krijgt een score", "c", "middle")
PICTURES["7.3"] = ("Elk oordeel wordt gemeten", g)

# 7.4 recall per class against a threshold sends each class to head or generalist
g = ""
classes74 = [
    ("strafrecht", 0.92),
    ("economie", 0.86),
    ("burgerrechten", 0.78),
    ("vakbonden", 0.58),
    ("privacy", 0.40),
    ("belasting", 0.24),
]
thr = 110 + 0.7 * 170
for i, (name, rec) in enumerate(classes74):
    y = 28 + i * 26
    ok = rec >= 0.7
    g += (
        t(14, y + 14, name, "c")
        + l7_rect(110, y, rec * 170, 18, PT if ok else VT, P if ok else V)
        + t(292, y + 14, "head" if ok else "generalist", "l p" if ok else "l v")
    )
g += line(f"M {thr} 22 L {thr} 182", V, 2, "5 4")
g += t(110, 196, "recall", "c") + t(thr, 196, "drempel", "m v", "middle")
g += t(200, 216, "de recall per klasse kiest de route", "c", "middle")
PICTURES["7.4"] = ("Recall kiest de route", g)

# 7.5 cascade: head, router with a threshold knob, easy cases stay, hard ones go on
g = (
    box(14, 86, 54, 28, "item", "g")
    + arrow(70, 100, 94, 100)
    + box(98, 84, 64, 32, "head", "p", PT)
    + t(130, 136, "goedkoop", "m", "middle")
    + arrow(164, 100, 188, 100)
    + box(192, 82, 76, 36, "router", "v", VT)
)
g += (
    path("M 268 92 L 300 60")
    + box(302, 40, 82, 30, "antwoord", "g")
    + t(343, 32, "makkelijk", "c", "middle")
)
g += (
    path("M 268 108 L 300 140")
    + box(302, 130, 82, 30, "generalist", "p", PT)
    + t(343, 178, "moeilijk", "c", "middle")
)
g += (
    line("M 230 120 L 230 144", MUT, 1, "3 3")
    + line("M 200 150 L 260 150", MUT, 2)
    + f'<circle cx="242" cy="150" r="6" fill="{VT}" stroke="{V}" stroke-width="2"/>'
    + t(230, 174, "drempel", "l v", "middle")
)
g += t(200, 210, "de drempel stel je af tegen echte kosten", "c", "middle")
PICTURES["7.5"] = ("Cascade met een drempel", g)

# 7.6 nearest labelled neighbours in the cache go to the generalist or straight to recall@k
g = t(20, 34, "cache uit les 1", "l p")
for x, y in ((50, 62), (78, 52), (82, 86), (108, 68), (60, 92)):
    g += dot(x, y)
for x, y in ((40, 150), (65, 140), (70, 172), (95, 158), (50, 178)):
    g += f'<rect x="{x - 5}" y="{y - 5}" width="10" height="10" fill="{BG}" stroke="{P}" stroke-width="1.5"/>'
for x, y in ((82, 86), (108, 68), (60, 92)):
    g += (
        line(f"M 100 98 L {x} {y}", V, 1.5)
        + f'<circle cx="{x}" cy="{y}" r="9" fill="none" stroke="{V}" stroke-width="2"/>'
    )
g += dot(100, 98, 7, VT, V) + t(100, 124, "nieuw", "l v", "middle")
g += (
    arrow(128, 92, 232, 69, "v", 2)
    + t(178, 66, "retrieval", "l v", "middle")
    + box(236, 50, 120, 36, "generalist", "p", PT)
    + t(296, 104, "duur per aanroep", "m", "middle")
)
g += (
    arrow(128, 110, 232, 150)
    + box(236, 134, 120, 36, "recall@k", "v", VT)
    + t(296, 188, "zonder aanroep", "m", "middle")
)
g += t(200, 212, "het geheugen meet je los van de generalist", "c", "middle")
PICTURES["7.6"] = ("De cache is een geheugen", g)

# 7.7 one test set, two models, read in two columns: score close, cost far apart
g = t(250, 34, "score", "l", "middle") + t(345, 34, "kosten", "l", "middle")
g += (
    box(14, 82, 70, 50, "testset", "g")
    + path("M 86 100 L 108 72")
    + path("M 86 114 L 108 144")
    + box(110, 56, 96, 28, "head", "p", PT)
    + box(110, 132, 96, 28, "generalist", "p", PT)
)
g += (
    l7_rect(220, 62, 46, 16, PT, P)
    + l7_rect(220, 138, 54, 16, PT, P)
    + l7_rect(310, 62, 3, 16, PT, P)
    + l7_rect(310, 138, 70, 16, PT, P)
)
g += t(250, 112, "Δ klein", "c v", "middle") + t(
    345, 112, "veel duurder", "c v", "middle"
)
g += t(200, 205, "zelfde testset: score én kosten naast elkaar", "c", "middle")
PICTURES["7.7"] = ("Head tegen generalist", g)

# 7.8 score against cost: the router curve above the random-routing control, at the budget
g = line("M 50 26 L 50 176 L 376 176", MUT, 1) + t(56, 34, "score", "c")
g += t(376, 194, "kosten", "c", "end")
g += line("M 60 140 L 360 60", MUT, 1.5, "5 4") + t(
    240, 112, "control: willekeurig", "c"
)
g += line("M 60 140 C 120 70, 200 62, 360 60", V, 2.5) + t(
    205, 52, "router", "l v", "middle"
)
g += dot(60, 140, 5, BG, P) + t(62, 160, "alleen head", "m")
g += dot(360, 60, 5, BG, P) + t(376, 40, "alleen generalist", "m", "end")
g += line("M 150 176 L 150 116", RULE, 1, "4 3") + line("M 150 116 L 150 84", V, 3)
g += dot(150, 81, 5, VT, V) + t(150, 194, "budget → drempel", "m v", "middle")
g += t(200, 214, "de afstand tot de control is wat de router oplevert", "c", "middle")
PICTURES["7.8"] = ("De router tegen een control", g)

# 7.9 score per number of retrieved neighbours, k = 0 being zero-shot
g = line("M 50 30 L 50 170 L 376 170", MUT, 1) + t(56, 34, "score", "c")
pts79 = [(0, 80, 130), (1, 140, 104), (3, 200, 88), (5, 260, 82), (10, 320, 80)]
for k, x, y in pts79:
    col, fill = (P, PT) if k == 0 else (V, VT)
    g += interval(x, y, 10, col, fill) + t(x, 188, str(k), "m", "middle")
g += t(80, 112, "zero-shot", "l p", "middle")
g += line("M 90 130 L 370 130", RULE, 1, "4 3") + arrow(360, 128, 360, 84, "v", 2)
g += t(354, 112, "winst", "l v", "end")
g += t(200, 210, "aantal meegegeven buren (k)", "c", "middle")
PICTURES["7.9"] = ("Meten wat geheugen oplevert", g)

# 7.10 published class runs of three kinds flow into one run that names its sources
g = l7_rect(14, 22, 372, 70, BG, RULE, 1) + t(24, 40, "klas-runs op SCOTUS", "c")
for x, lab in ((24, "head"), (150, "generalist"), (276, "retrieval")):
    g += box(x, 50, 100, 32, lab, "p", PT)
g += (
    path("M 74 84 L 150 132", "v", 2)
    + path("M 200 84 L 200 132", "v", 2)
    + path("M 326 84 L 250 132", "v", 2)
)
g += box(110, 136, 180, 32, "jouw cascade", "v", VT) + t(
    200, 186, "builds_on: 3 bronnen", "m", "middle"
)
g += t(200, 212, "publiceren, bekijken, voortbouwen met bron", "c", "middle")
PICTURES["7.10"] = ("Als klas combineren", g)


def goals(block: str, lesson: int) -> list[tuple[str, str]]:
    out = []
    for paragraph in block.strip().split("\n\n"):
        m = re.match(rf"({lesson}\.\d+) (.*)", " ".join(paragraph.split()))
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def inline(text: str) -> str:
    coded = re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(text))
    return re.sub(r"\*([^*]+)\*", r"<em>\1</em>", coded)


# What a picture leaves out of its goal, shown under the picture.
NOT_SHOWN: dict[str, str] = {
    "1.11": "de geschiedenis van AI",
    "2.4": "wat downsampling met het receptive field doet, en kernels van verschillende grootte naast elkaar",
    "2.5": "wat stap voor stap rekenen kost",
    "2.11": "de metric die bij het probleem past",
    "3.4": "hoe specialisten en een generalist samen in één netwerk zitten",
    "3.9": "dat een vlak search-resultaat informatie is en geen mislukking",
    "4.5": "waarom hele woorden als tokens tekortschieten: een te grote vocabulary en onbekende woorden",
    "4.12": "dat er per klasse weinig voorbeelden zijn",
    "5.7": "de target encoder die het doel in de latente ruimte levert",
    "6.12": "het publiceren, bekijken en voortbouwen als klas; dat beeld staat bij 2.12",
    "7.5": "de control waartegen de router zijn plek moet verdienen; die staat bij 7.8",
}


def card(code: str, text: str) -> str:
    title, svg = PICTURES[code]
    note = (
        f'\n  <p class="note">Let op: dit beeld toont niet {NOT_SHOWN[code]}.</p>'
        if code in NOT_SHOWN
        else ""
    )
    return f"""<article class="goal" id="g{code.replace(".", "-")}">
  <header><span class="code">{code}</span><h3>{html.escape(title)}</h3></header>
  <svg viewBox="0 0 400 230" role="img" aria-label="{html.escape(title)}">{svg}</svg>{note}
  <p>{inline(text)}</p>
</article>"""


def marker(c: str, col: str) -> str:
    return (
        f'<marker id="a{c}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{col}"/></marker>'
    )


def build(lesson: int) -> Path:
    md = (ROOT / f"references/leerdoelen/les{lesson}.md").read_text()
    heading = re.match(r"# Les \d+ — (.*)", md)
    if heading is None:
        raise SystemExit(
            f"les{lesson}.md does not start with '# Les {lesson} — <title>'"
        )
    title = html.escape(heading.group(1))
    body = md.split("## De student begrijpt")[1].split("## Materiaal")[0]
    und, can = (goals(block, lesson) for block in body.split("## De student kan"))
    missing = [code for code, _ in und + can if code not in PICTURES]
    if missing:
        raise SystemExit(f"no picture for {', '.join(missing)}: add them to PICTURES")
    n = len(und) + len(can)
    understands = "\n".join(card(code, text) for code, text in und)
    able = "\n".join(card(code, text) for code, text in can)
    markers = marker("g", MUT) + marker("v", V) + marker("p", P)
    page = f"""<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MADS-ML · Les {lesson} · Leerdoelen in beeld</title>
<meta name="description" content="De leerdoelen van les {lesson} van MADS-ML, elk met één beeld.">
<style>
@font-face{{font-family:'JetBrains Mono';font-weight:400;font-display:swap;src:url(fonts/JetBrainsMono-400.woff2) format('woff2')}}
@font-face{{font-family:'Outfit';font-weight:400;font-display:swap;src:url(fonts/Outfit-400.woff2) format('woff2')}}
@font-face{{font-family:'Outfit';font-weight:500;font-display:swap;src:url(fonts/Outfit-500.woff2) format('woff2')}}
:root{{color-scheme:light}}
*{{box-sizing:border-box}}
html,body{{margin:0;background:{BG};color:{INK}}}
body{{font-family:Outfit,Arial,sans-serif;font-size:17px;line-height:1.45}}
main{{max-width:1240px;margin:0 auto;padding:40px 16px 64px}}
.kick{{font-family:'JetBrains Mono',monospace;font-size:13px;letter-spacing:2.4px;text-transform:uppercase;color:{P};margin:0 0 14px}}
h1{{font-size:clamp(34px,6vw,60px);font-weight:500;line-height:1.05;letter-spacing:-1.2px;margin:0 0 16px}}
h1 span{{color:{V}}}
.lead{{max-width:720px;color:{MUT};margin:0 0 20px}}
.legend{{display:flex;flex-wrap:wrap;gap:8px 22px;font-size:15px;color:{MUT};margin:0 0 8px}}
.legend i{{display:inline-block;width:14px;height:14px;margin-right:7px;vertical-align:-2px;border:2px solid}}
h2{{font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:400;letter-spacing:2.4px;text-transform:uppercase;color:{MUT};margin:44px 0 16px;padding-top:16px;border-top:1px solid {RULE}}}
.grid{{display:grid;gap:20px;grid-template-columns:1fr}}
@media(min-width:760px){{.grid{{grid-template-columns:1fr 1fr}}}}
@media(min-width:1160px){{.grid{{grid-template-columns:1fr 1fr 1fr}}}}
.goal{{border:1px solid {RULE};padding:18px 18px 16px;display:flex;flex-direction:column;gap:10px;scroll-margin-top:16px}}
.goal:target{{border:2px solid {V}}}
.goal header{{display:flex;align-items:baseline;gap:12px}}
.code{{font-family:'JetBrains Mono',monospace;font-size:14px;color:{P}}}
h3{{font-size:21px;font-weight:500;line-height:1.2;margin:0}}
.goal svg{{width:100%;height:auto;display:block}}
.goal p{{margin:0;font-size:15.5px;color:{MUT}}}
.goal p.note{{font-size:14px;color:{V};padding-left:10px;border-left:2px solid {V}}}
code{{font-family:'JetBrains Mono',monospace;font-size:.88em;color:{P}}}
svg text{{font-family:Outfit,Arial,sans-serif;fill:{INK}}}
svg .l{{font-size:14px;font-weight:500}}
svg .c{{font-size:14px;fill:{MUT}}}
svg .m{{font-family:'JetBrains Mono',monospace;font-size:12.5px;fill:{MUT}}}
svg .n{{font-family:'JetBrains Mono',monospace;font-size:10.5px;fill:{MUT}}}
svg .big{{font-size:20px;font-weight:500}}
svg .v{{fill:{V}}}
svg .p{{fill:{P}}}
svg .n.v{{fill:{V}}}
</style>
</head>
<body>
<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>{markers}</defs></svg>
<main>
<p class="kick">MADS-ML · Les {lesson}</p>
<h1>{title}</h1>
<p class="lead">{n} leerdoelen, elk met één beeld. Het beeld draagt het punt; de tekst eronder is het leerdoel zoals je het afvinkt.</p>
<p class="legend"><span><i style="border-color:{P};background:{PT}"></i>de data: een tensor en zijn assen</span><span><i style="border-color:{V};background:{VT}"></i>de operatie: wat er nu iets doet</span></p>
<h2>De student begrijpt</h2>
<div class="grid">
{understands}
</div>
<h2>De student kan</h2>
<div class="grid">
{able}
</div>
</main>
</body>
</html>
"""
    out = ROOT / f"presentations/leerdoelen-les{lesson}.html"
    out.write_text(page)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One picture per learning goal of a lesson."
    )
    parser.add_argument("--lesson", type=int, required=True)
    print(build(parser.parse_args().lesson))


if __name__ == "__main__":
    main()
