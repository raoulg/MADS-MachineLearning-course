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
    + t(323, 72, "mlflow", "l p", "middle")
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
