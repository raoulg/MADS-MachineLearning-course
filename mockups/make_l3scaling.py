import math
from pathlib import Path

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
# plot box: x is parameters x1..x8, y is data x1..x8, every doubling one step
X0, Y0, X1, Y1 = 400, 462, 1240, 90


def px(f):
    return X0 + (X1 - X0) * math.log2(f) / 3


def py(f):
    return Y0 - (Y0 - Y1) * math.log2(f) / 3


def txt(x, y, s, size=30, fill=MUT, anchor="start", weight=400):
    return f'<text x="{x:.0f}" y="{y:.0f}" font-family="Outfit,Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>'


def axes():
    out = (
        f'<line x1="{X0}" y1="{Y0}" x2="{X1 + 60}" y2="{Y0}" stroke="{MUT}" stroke-width="2.5" marker-end="url(#ag)"/>'
        f'<line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1 - 30}" stroke="{MUT}" stroke-width="2.5" marker-end="url(#ag)"/>'
        + txt(X1 + 60, Y0 + 74, "parameters →", 32, MUT, "end")
        + txt(X0 - 30, Y1 - 44, "data ↑", 32, MUT, "end")
    )
    for f in (1, 2, 4, 8):
        out += f'<line x1="{px(f):.0f}" y1="{Y0}" x2="{px(f):.0f}" y2="{Y0 + 12}" stroke="{MUT}" stroke-width="2"/>'
        out += txt(px(f), Y0 + 44, f"×{f}", 28, MUT, "middle")
        out += f'<line x1="{X0 - 12}" y1="{py(f):.0f}" x2="{X0}" y2="{py(f):.0f}" stroke="{MUT}" stroke-width="2"/>'
        out += txt(X0 - 22, py(f) + 10, f"×{f}", 28, MUT, "end")
    return out


def dot(f_n, f_d, r=16, fill=BG, stroke=INK, sw=4):
    return f'<circle cx="{px(f_n):.0f}" cy="{py(f_d):.0f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def path(exponent, colour, width=5, dash=""):
    pts = [(px(2 ** (3 * i / 40)), py(2 ** (3 * i / 40 * exponent))) for i in range(41)]
    d = "M " + " L ".join(f"{x:.0f} {y:.0f}" for x, y in pts)
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}"{da}/>'


def arrow(x0, y0, x1, y1, colour, marker):
    return f'<line x1="{x0:.0f}" y1="{y0:.0f}" x2="{x1:.0f}" y2="{y1:.0f}" stroke="{colour}" stroke-width="4" marker-end="url(#{marker})"/>'


KAPLAN_PENALTY = (
    0.076 / 0.103
)  # D ~ N^0.74 keeps the overfitting penalty equal (eq. 1.5)
KAPLAN_SPLIT = 0.27 / 0.73  # N ~ C^0.73, D ~ C^0.27
HOFFMANN_SPLIT = 0.50 / 0.50  # N ~ C^0.50, D ~ C^0.50
k_pen = 8**KAPLAN_PENALTY
k_split = 8**KAPLAN_SPLIT

# slide 1: how much data for 8x the parameters, Kaplan's equal penalty
s1 = axes()
s1 += (
    "<g data-build-in>"
    + arrow(px(1) + 20, py(1), px(8) - 22, py(1), INK, "ak")
    + dot(8, 1, 15, BG, MUT, 4)
    + txt(px(8) - 24, py(1) - 22, "too little data", 30, MUT, "end", 500)
    + "</g>"
)
s1 += (
    "<g data-build-in>"
    + path(KAPLAN_PENALTY, P, 6)
    + arrow(px(8), py(1) - 22, px(8), py(k_pen) + 24, P, "ap2")
    + dot(8, k_pen, 16, PT, P, 4)
    + txt(px(8) + 30, py(k_pen) + 10, "about ×5 data", 32, P, "start", 600)
    + "</g>"
)
s1 += dot(1, 1, 16, BG, INK, 4)

# slide 2: Gopher, six sizes on the same 300B tokens (Rae et al. 2021, Table 4, RACE-h)
GOPHER = [("417M", 27.2), ("1.4B", 26.0), ("7.1B", 30.6), ("280B", 71.6)]
GB0, GTOP, GMAX = 470, 70, 80  # baseline y, top y, score at the top


def gy(v):
    return GB0 - (GB0 - GTOP) * v / GMAX


s2 = (
    f'<line x1="{X0}" y1="{GB0}" x2="{X1 + 40}" y2="{GB0}" stroke="{MUT}" stroke-width="2.5"/>'
    + txt(X0 - 30, GTOP - 30, "reading comprehension score ↑", 30, MUT, "start")
)
for v in (0, 25, 50, 75):
    s2 += (
        f'<line x1="{X0 - 12}" y1="{gy(v):.0f}" x2="{X0}" y2="{gy(v):.0f}" stroke="{MUT}" stroke-width="2"/>'
        + txt(X0 - 22, gy(v) + 10, f"{v}%", 28, MUT, "end")
    )
s2 += (
    f'<line x1="{X0}" y1="{gy(25):.0f}" x2="{X1 + 260}" y2="{gy(25):.0f}" stroke="{MUT}" stroke-width="2.5" stroke-dasharray="10 8"/>'
    + txt(X1 + 260, gy(25) - 14, "guessing: 1 in 4", 28, MUT, "end", 500)
)
for i, (size, score) in enumerate(GOPHER):
    cx = X0 + 130 + i * 215
    bar = (
        f'<rect x="{cx - 70}" y="{gy(score):.0f}" width="140" height="{GB0 - gy(score):.0f}" fill="{P}"/>'
        + txt(cx, gy(score) - 14, f"{score:.0f}%", 30, P, "middle", 600)
        + (
            txt(cx, gy(score) + 40, "Gopher", 28, "#ffffff", "middle", 600)
            if size == "280B"
            else ""
        )
        + txt(cx, GB0 + 40, size, 30, INK, "middle", 600)
        + txt(cx, GB0 + 74, "300B tokens", 26, MUT, "middle")
    )
    s2 += f"<g data-build-in>{bar}</g>" if size == "280B" else bar
s2 += txt(X0 - 30, GB0 + 40, "parameters", 30, INK, "end", 600) + txt(
    X0 - 30, GB0 + 74, "data", 26, MUT, "end"
)

# slide 3: Chinchilla against Gopher, one row per quantity, each row on its own scale
ROWS = [
    ("parameters", 280, 70, "280B", "70B", "4× fewer"),
    ("data", 300, 1400, "300B tokens", "1.4T tokens", "4.7× more"),
    ("knowledge test", 60.0, 67.6, "60%", "68%", "higher"),
]
BX, BW, BH = 360, 900, 46


def bars(y, gv, cv, gl, cl, top, factor, first):
    w = lambda v: BW * v / top
    gname = txt(BX + 16, y + 33, "Gopher", 26, "#ffffff", "start", 600) if first else ""
    cname = (
        txt(BX + 16, y + BH + 12 + 33, "Chinchilla", 26, "#ffffff", "start", 600)
        if first
        else ""
    )
    g = (
        f'<rect x="{BX}" y="{y}" width="{w(gv):.0f}" height="{BH}" fill="{P}"/>'
        + gname
        + txt(BX + w(gv) + 16, y + 34, gl, 30, P, "start", 600)
    )
    c = (
        f'<rect x="{BX}" y="{y + BH + 12}" width="{w(cv):.0f}" height="{BH}" fill="{V}"/>'
        + cname
        + txt(BX + w(cv) + 16, y + BH + 12 + 34, cl, 30, V, "start", 600)
        + txt(1700, y + BH + 18, factor, 36, V, "end", 700)
    )
    return g, c


s3, s3c = "", ""
for r, (name, gv, cv, gl, cl, factor) in enumerate(ROWS):
    y = 40 + r * 175
    top = 100 if name == "knowledge test" else max(gv, cv)
    g, c = bars(y, gv, cv, gl, cl, top, factor, r == 0)
    s3 += (
        txt(BX - 30, y + BH + 16, name, 34, INK, "end", 600)
        + f'<line x1="{BX}" y1="{y - 10}" x2="{BX}" y2="{y + 2 * BH + 22}" stroke="{MUT}" stroke-width="2.5"/>'
        + g
    )
    s3c += c
s3 += f"<g data-build-in>{s3c}</g>"

MARKERS = (
    f'<defs><marker id="ak" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
    f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/></marker>'
    f'<marker id="ap2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
    f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{P}"/></marker></defs>'
)


def section(sid, crumb, title, svg, sentence, credit=""):
    c = f'\n    <p class="credit">{credit}</p>' if credit else ""
    cr = f' data-crumb="{crumb}"' if crumb else ""
    return f'''<section id="{sid}" class="paper std"{cr}>
  <img class="strip" src="img/lesson1/band_strip_paper.png" alt="pttrn logo in a Rule 30 strip">
  <div class="kick"><p>Lesson 3</p><p>Scaling laws</p></div>
  <div class="body" style="gap:20px">
    <h2 class="t">{title}</h2>
    <svg viewBox="0 0 1720 560" width="1720" height="560" role="img" aria-label="{title}">{MARKERS}{svg}</svg>
    <p class="s" data-build-in>{sentence}</p>{c}
  </div>
  <p class="pn n"></p>
</section>
'''


body = section(
    "data-for-parameters",
    "01 Parameters and data",
    "A model 8× bigger: how much more data?",
    s1,
    "According to Kaplan (2020), models should scale their data with about 5/8 of the parameters to avoid overfitting.",
    "Kaplan et al. (2020), Scaling laws for neural language models · language models.",
)
body += section(
    "gopher",
    "",
    "Gopher: more parameters on the same data, a better model",
    s2,
    "DeepMind trained six sizes of one model, all on the same 300B tokens. The largest, Gopher, was far better than the rest.",
    "Rae et al. (2021), Scaling language models: methods, analysis &amp; insights from training Gopher · RACE-h, Table 4.",
)
body += section(
    "chinchilla",
    "",
    "Chinchilla: the same compute as Gopher, spent on data",
    s3,
    "Chinchilla used the compute budget of Gopher for a model 4× smaller, trained on 4.7× more data, and scored higher.",
    "Hoffmann et al. (2022), Training compute-optimal large language models · knowledge test: MMLU, Table 6.",
)
src = Path("lesson3-tradeoffs.html").read_text()
head = src[: src.index("<section")].replace(
    "Lesson 3 · Weighing tradeoffs", "Lesson 3 · Scaling laws"
)
tail = src[src.rindex("</section>") + len("</section>\n") :]
Path("lesson3-scaling.html").write_text(head + "\n" + body + tail)
print(f"Kaplan penalty x{k_pen:.2f}, Kaplan split x{k_split:.2f}")
