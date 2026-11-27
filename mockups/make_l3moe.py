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
GREY, GREYT = "#8a837a", "#e4ded5"
S, G = 26, 32  # parameter block: side and pitch
UNIT = 1_000  # parameters per block
HERE = Path(__file__).parent
TEMPLATE = HERE / "lesson3-tradeoffs.html"  # the deck frame of the lesson 3 mockups

# Round numbers for the concept: a dense network of 10,000 parameters, and a MoE of six experts
# whose router and one expert together use as many. Notebook 3.2's MoE 8x8 on mix7 stores
# 27,792 parameters and uses 3,929.
N_EXPERTS = 6
DENSE, ROUTER, EXPERT = 10_000, 1_000, 9_000
TOTAL, ACTIVE = ROUTER + N_EXPERTS * EXPERT, ROUTER + EXPERT
assert (TOTAL, ACTIVE) == (55_000, 10_000)


def n(v):
    return f"{v:,}"


def txt(x, y, s, size=30, fill=MUT, anchor="start", weight=400):
    return f'<text x="{x:.0f}" y="{y:.0f}" font-family="Outfit,Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>'


def sq(x, y, kind):
    fill, stroke = {
        "used": (VT, V),
        "idle": (GREYT, GREY),
        "router": (PT, P),
        "unit": ("none", MUT),
    }[kind]
    return f'<rect x="{x}" y="{y}" width="{S}" height="{S}" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>'


def block(x0, y0, count, kind, cols=2, start=0):
    """count blocks, `cols` wide, filled row by row from position `start`."""
    return "".join(
        sq(x0 + (i % cols) * G, y0 + (i // cols) * G, kind)
        for i in range(start, start + count)
    )


def rect(x, y, w, h, fill, stroke, rx=6):
    return f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>'


def arrow(x1, y1, x2, y2, colour, marker, width=3, path=None):
    d = path or f"M {x1} {y1} L {x2} {y2}"
    return f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}" marker-end="url(#{marker})"/>'


assert [v // UNIT for v in (DENSE, ROUTER, EXPERT)] == [10, 1, 9]
EX = [430 + e * 150 for e in range(N_EXPERTS)]  # left edge of each expert's column

# slide 1: parameters stored against parameters used for one image. Every block is two wide,
# so the router stacked on one expert is the same shape as the dense network.
s1 = txt(0, 30, "Stored", 32, INK, weight=500)
s1 += txt(0, 150, "Dense network", 32, INK, weight=500) + block(EX[0], 60, 10, "used")
s1 += txt(EX[0] + 90, 150, f"{n(DENSE)} parameters, all used for every image", 30)
s1 += txt(0, 330, f"MoE, {N_EXPERTS} experts", 32, INK, weight=500) + block(
    300, 306, 1, "router"
)
s1 += txt(313, 290, "router", 28, P, "middle")
for e, x in enumerate(EX):
    s1 += block(x, 250, 9, "used" if e == 2 else "idle") + txt(
        x + 29, 448, f"E{e + 1}", 28, MUT, "middle"
    )
s1 += txt(300, 490, f"{n(TOTAL)} parameters stored, {n(ACTIVE)} used for one image", 30)
s1 += f'<line x1="1290" y1="0" x2="1290" y2="470" stroke="{RULE}" stroke-width="2"/>'
s1 += txt(1330, 30, "Used for one image", 32, INK, weight=500)
s1 += block(1350, 70, 10, "used") + txt(1379, 270, n(DENSE), 30, MUT, "middle")
s1 += txt(1490, 160, "=", 44, INK, "middle")
s1 += block(1570, 70, 1, "router") + block(1570, 70, 9, "used", start=1)
s1 += txt(1599, 270, n(ACTIVE), 30, MUT, "middle")
s1 += sq(0, 520, "used") + txt(40, 542, "used for this image", 28)
s1 += sq(330, 520, "idle") + txt(370, 542, "stored, idle for this image", 28)
s1 += sq(750, 520, "router") + txt(790, 542, "router, used for every image", 28)
s1 += sq(1200, 520, "unit") + txt(1240, 542, f"= {n(UNIT)} parameters", 28)

# slide 2: all training images (dots) to one network, or split by the router over the experts, unevenly
COUNTS = [13, 8, 11, 9, 12, 7]
PITCH, ROWS3 = 22, 3


def dots(cx, y0, count, rows=ROWS3, fill=INK):
    cols = -(-count // rows)
    x0 = cx - (cols - 1) * PITCH / 2
    return "".join(
        f'<circle cx="{x0 + (i // rows) * PITCH:.0f}" cy="{y0 + (i % rows) * PITCH:.0f}" r="7" fill="{fill}"/>'
        for i in range(count)
    )


ECX = [400 + e * 200 for e in range(N_EXPERTS)]
s2 = txt(0, 100, "Dense network", 32, INK, weight=500)
s2 += dots(900, 50, sum(COUNTS)) + txt(
    900 + 11 * PITCH, 82, "all training images", 30, MUT
)
s2 += arrow(900, 112, 900, 152, MUT, "ag")
s2 += rect(700, 156, 400, 56, "#ffffff", MUT) + txt(
    900, 194, "one network learns from all", 30, INK, "middle"
)
s2 += txt(0, 330, f"MoE, {N_EXPERTS} experts", 32, INK, weight=500)
for cx, c in zip(ECX, COUNTS):
    s2 += dots(cx, 280, c)
    s2 += arrow(cx, 342, cx, 382, MUT, "ag")
    s2 += rect(cx - 50, 386, 100, 56, "#ffffff", MUT) + txt(
        cx, 424, f"E{ECX.index(cx) + 1}", 28, INK, "middle"
    )
s2 += txt(ECX[0] - 50, 500, "the router splits them, not evenly", 30, MUT)

# slide 3: router collapse as the shares of the images over six experts, then the same with a
# balancing loss. Schematic, after Shazeer et al. (2017) and Fedus, Zoph and Shazeer (2022).


def shares(x0, base, heights, colour, fill, label):
    out = txt(x0, base - 330, label, 32, INK, weight=600)
    for e, h in enumerate(heights):
        out += rect(x0 + e * 110, base - h, 76, h, fill, colour, 3) + txt(
            x0 + e * 110 + 38, base + 36, f"E{e + 1}", 26, MUT, "middle"
        )
    return (
        out
        + f'<line x1="{x0 - 14}" y1="{base}" x2="{x0 + 6 * 110}" y2="{base}" stroke="{MUT}" stroke-width="2"/>'
    )


s3 = shares(
    60,
    440,
    [6, 6, 290, 6, 6, 6],
    V,
    VT,
    "without balancing loss: one expert gets everything",
)
s3 += txt(60, 520, "share of the images per expert", 26, MUT)
s3 += f'<line x1="860" y1="60" x2="860" y2="500" stroke="{RULE}" stroke-width="2"/>'
s3 += (
    "<g data-build-in>"
    + shares(
        960,
        440,
        [44, 54, 48, 40, 52, 46],
        P,
        PT,
        "with balancing loss: every expert gets images",
    )
    + "</g>"
)

# slide 4: time per image on one scale for both situations. A dense network that stores as much
# runs all six experts' worth; a MoE runs one expert plus a routing cost (router, sort, gather,
# scatter) that hardly grows with the experts. Schematic; notebook 3.2 measured the small end.
ROUTING = 6
TBX, TU = 400, 19  # bar start, pixels per unit of time


def tbar(y, segments):
    out, x = "", TBX
    for units, fill, stroke in segments:
        out += rect(x, y, units * TU, 50, fill, stroke, 3)
        x += units * TU
    return out, x


def situation(y, title, expert):
    out = txt(0, y + 4, title, 32, INK, weight=600)
    dense, dx = tbar(y + 30, [(expert, VT, V)] * N_EXPERTS)
    moe, mx = tbar(y + 100, [(ROUTING, PT, P), (expert, VT, V)])
    out += txt(0, y + 66, "dense, stores as much", 28, INK) + dense
    out += txt(0, y + 136, "MoE: routing + 1 expert", 28, INK) + moe
    faster = mx < dx
    out += txt(
        mx + 20,
        y + 136,
        "faster" if faster else "slower",
        30,
        P if faster else V,
        "start",
        700,
    )
    return out


s4 = situation(40, "Small experts", 1) + situation(250, "Large experts", 10)
s4 += f'<line x1="{TBX}" y1="465" x2="{TBX + 62 * TU}" y2="465" stroke="{MUT}" stroke-width="2" marker-end="url(#ag)"/>'
s4 += txt(TBX + 62 * TU, 498, "time per image", 26, MUT, "end")
s4 += rect(0, 515, 40, 30, VT, V, 3) + txt(56, 540, "one expert&#8217;s arithmetic", 28)
s4 += rect(470, 515, 40, 30, PT, P, 3) + txt(
    526, 540, "routing: router, sort, gather, scatter", 28
)


def section(sid, crumb, title, svg, sentence="", credit=""):
    cr = f' data-crumb="{crumb}"' if crumb else ""
    s = f'\n    <p class="s" data-build-in>{sentence}</p>' if sentence else ""
    c = f'\n    <p class="credit">{credit}</p>' if credit else ""
    return f'''<section id="{sid}" class="paper std"{cr}>
  <img class="strip" src="../pages/img/lesson1/band_strip_paper.png" alt="pttrn logo in a Rule 30 strip">
  <div class="kick"><p>Lesson 3</p><p>Mixture of experts</p></div>
  <div class="body" style="gap:20px">
    <h2 class="t">{title}</h2>
    <svg viewBox="0 0 1720 560" width="1720" height="560" role="img" aria-label="{title}">{svg}</svg>{s}{c}
  </div>
  <p class="pn n"></p>
</section>
'''


body = section(
    "moe-parameters",
    "01 Mixture of experts",
    "More parameters stored, the same compute per image",
    s1,
)
body += section("moe-data", "", "N experts learn from 1/Nth of the images", s2)
body += section(
    "moe-collapse",
    "",
    "Router collapse: one expert gets everything",
    s3,
    "The expert that starts a little better gets more images, learns faster from them, and gets even more. A balancing loss grows the more uneven the shares are, and so spreads the images again.",
    credit="Collapse: Shazeer et al. (2017), Outrageously large neural networks: the sparsely-gated mixture-of-experts layer · "
    "balancing loss: Fedus, Zoph &amp; Shazeer (2022), Switch Transformers",
)
body += section(
    "moe-crossover",
    "",
    "A MoE is faster only when routing is a small share",
    s4,
    "Routing costs about the same time whatever the size of an expert. Against small experts it outweighs the five experts a MoE skips; against large experts it is a small share.",
)
src = TEMPLATE.read_text()
head = src[: src.index("<section")].replace(
    "Lesson 3 · Weighing tradeoffs", "Lesson 3 · Mixture of experts"
)
tail = src[src.rindex("</section>") + len("</section>\n") :]
(HERE / "lesson3-moe.html").write_text(head + "\n" + body + tail)
