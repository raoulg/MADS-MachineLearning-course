import math
from pathlib import Path

P, PT, V, VT, MUT, RULE, BG = (
    "#1d5a73",
    "#c9dde4",
    "#c43a14",
    "#f3cdbf",
    "#5a544d",
    "#cfc7bb",
    "#f4f1ec",
)
X0, Y0, X1, Y1 = 220, 480, 1500, 40  # plot box: left, bottom, right, top


def txt(x, y, s, size=30, fill=MUT, anchor="start", weight=400, mono=False):
    fam = "'JetBrains Mono',monospace" if mono else "Outfit,Arial,sans-serif"
    return f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>'


def axes(xlab, ylab):
    return (
        f'<path d="M {X0} {Y1 - 10} L {X0} {Y0} L {X1 + 20} {Y0}" fill="none" stroke="{MUT}" stroke-width="2.5"/>'
        f'<line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1 - 14}" stroke="{MUT}" stroke-width="2.5" marker-end="url(#ag)"/>'
        f'<line x1="{X0}" y1="{Y0}" x2="{X1 + 24}" y2="{Y0}" stroke="{MUT}" stroke-width="2.5" marker-end="url(#ag)"/>'
        + txt(X1 + 20, Y0 + 50, xlab, 32, MUT, "end")
        + txt(X0 - 20, Y1 + 10, ylab, 32, MUT, "end")
    )


def dot(x, y, r=13, fill=PT, stroke=P, sw=3):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def quadrant(x, y, label):
    return (
        f'<rect x="{x}" y="{Y1}" width="{X1 - x}" height="{y - Y1}" fill="{VT}" fill-opacity="0.55" stroke="{V}" stroke-width="2" stroke-dasharray="8 6"/>'
        + txt(X1 - 16, Y1 + 40, label, 30, V, "end", 500)
    )


# slide 1: models on score and speed (more is better on both, so the best lie up and right)
pts = [
    (300, 440),
    (360, 390),
    (420, 330),
    (470, 180),
    (520, 420),
    (560, 290),
    (640, 140),
    (700, 350),
    (760, 240),
    (820, 400),
    (880, 200),
    (960, 300),
    (1020, 250),
    (1080, 380),
    (1140, 330),
    (1200, 300),
    (1280, 420),
    (1350, 390),
    (1420, 440),
]


def dominated(p):
    return any(q[0] >= p[0] and q[1] <= p[1] and q != p for q in pts)


front = sorted([p for p in pts if not dominated(p)])
A = (700, 350)
B = (880, 200)
F = front[2]
assert dominated(A) and B in pts and not dominated(F)


def box_right(x, y, fill=True):
    f = f'fill="{VT}" fill-opacity="0.55"' if fill else 'fill="none"'
    return f'<rect x="{x}" y="{Y1}" width="{X1 - x}" height="{y - Y1}" {f} stroke="{V}" stroke-width="2" stroke-dasharray="8 6"/>'


dots = "".join(dot(x, y) for x, y in pts)
cloud = axes("speed →", "score ↑") + dots
s1 = axes("speed →", "score ↑") + (
    "<g data-build-in>"
    + box_right(A[0], A[1])
    + txt(X1 - 16, Y1 + 40, "better on both", 30, V, "end", 500)
    + dot(*A, 16, BG, P, 5)
    + txt(A[0] - 24, A[1] + 50, "this model", 30, P, "end", 500)
    + f'<line x1="{A[0] + 12}" y1="{A[1] - 12}" x2="{B[0] - 14}" y2="{B[1] + 14}" stroke="{V}" stroke-width="4" marker-end="url(#av)"/>'
    + dot(*B, 16, VT, V, 4)
    + txt(B[0] + 26, B[1] - 22, "faster and higher", 30, V, "start", 500)
    + "</g>"
)
s1 += dots
s1 = (
    s1.replace(dot(*A, 16, BG, P, 5), "").replace(dot(*B, 16, VT, V, 4), "")
    + dot(*A, 16, BG, P, 5)
    + dot(*B, 16, VT, V, 4)
)
s1b = cloud + (
    "<g data-build-in>"
    + box_right(F[0], F[1], False)
    + dot(*F, 16, VT, V, 5)
    + txt(X1 - 16, Y1 + 40, "empty:", 30, V, "end", 500)
    + txt(X1 - 16, Y1 + 80, "nothing beats it on both", 30, V, "end", 500)
    + "</g>"
)
fd = "M " + " L ".join(f"{x} {y}" for x, y in front)
s1b += (
    f'<g data-build-in><path d="{fd}" fill="none" stroke="{V}" stroke-width="5"/>'
    + "".join(dot(x, y, 15, V, V) for x, y in front)
    + txt(front[4][0] + 24, front[4][1] - 34, "Pareto front", 32, V, "start", 600)
    + "</g>"
)
top = min(front, key=lambda p: p[1])
a0 = (front[2][0] - 24, front[2][1] - 44)
a1 = (top[0] + 30, top[1] - 40)
s1b += (
    "<g data-build-in>"
    + f'<circle cx="{top[0]}" cy="{top[1]}" r="26" fill="none" stroke="{P}" stroke-width="3"/>'
    + txt(top[0] - 36, top[1] - 30, "best on score alone", 30, P, "end", 500)
    + f'<path d="M {a0[0]} {a0[1]} L {a1[0]} {a1[1]}" fill="none" stroke="{P}" stroke-width="3" stroke-dasharray="8 6" marker-end="url(#ap)"/>'
    + f'<g transform="rotate({math.degrees(math.atan2(a0[1] - a1[1], a0[0] - a1[0])):.1f} {(a0[0] + a1[0]) / 2} {(a0[1] + a1[1]) / 2 - 16})">'
    + txt(
        (a0[0] + a1[0]) / 2,
        (a0[1] + a1[1]) / 2 - 16,
        "slower, higher score",
        28,
        P,
        "middle",
        500,
    )
    + "</g></g>"
)


# slide 2: admissions, after Nilforoshan et al. (2022), schematic
def fx(u):  # front: graduates against admitted from target group, u in [0,1]
    return X0 + 120 + u * 1080, Y1 + 50 + 330 * u**2.2


curve = [fx(i / 40) for i in range(41)]
cd = "M " + " L ".join(f"{x:.0f} {y:.0f}" for x, y in curve)
C = (760, 330)
s2 = axes("admitted from the target group →", "graduates ↑")
s2 += (
    f'<path d="{cd} L {curve[-1][0]:.0f} {Y0} L {X0 + 120} {Y0} Z" fill="{PT}" fill-opacity="0.35" stroke="none"/>'
    f'<path d="{cd}" fill="none" stroke="{V}" stroke-width="5"/>'
    + txt(curve[4][0] + 10, curve[4][1] - 26, "Pareto front", 32, V, "start", 600)
    + txt(X0 + 160, Y0 - 30, "possible policies", 28, P)
)
# part of the front that beats C on both
seg = [p for p in curve if p[0] >= C[0] and p[1] <= C[1]]
sd = "M " + " L ".join(f"{x:.0f} {y:.0f}" for x, y in seg)
s2 += (
    "<g data-build-in>"
    + dot(*C, 17, BG, P, 5)
    + txt(C[0] - 30, C[1] + 56, "policy under the rule", 30, P, "end", 500)
    + "</g>"
)
s2 += (
    f'<g data-build-in><rect x="{C[0]}" y="{Y1}" width="{X1 - C[0]}" height="{C[1] - Y1}" fill="{VT}" fill-opacity="0.45" stroke="{V}" stroke-width="2" stroke-dasharray="8 6"/>'
    + dot(*C, 17, BG, P, 5)
    + f'<path d="{sd}" fill="none" stroke="{V}" stroke-width="12" stroke-linecap="round"/>'
    + txt(X1 - 16, Y1 + 40, "more admitted and more graduates", 30, V, "end", 500)
    + "</g>"
)


def section(sid, crumb, title, svg, sentence, credit=""):
    c = f'\n    <p class="credit">{credit}</p>' if credit else ""
    cr = f' data-crumb="{crumb}"' if crumb else ""
    return f'''<section id="{sid}" class="paper std"{cr}>
  <img class="strip" src="img/lesson1/band_strip_paper.png" alt="pttrn logo in a Rule 30 strip">
  <div class="kick"><p>Lesson 3</p><p>Weighing tradeoffs</p></div>
  <div class="body" style="gap:20px">
    <h2 class="t">{title}</h2>
    <svg viewBox="0 0 1720 560" width="1720" height="560" role="img" aria-label="{title}">{svg}</svg>
    <p class="s" data-build-in>{sentence}</p>{c}
  </div>
  <p class="pn n"></p>
</section>
'''


body = section(
    "pareto-beaten",
    "01 Pareto front",
    "Which model is best, if you care about two things?",
    s1,
    "This model is beaten: another model is faster and scores higher. The Pareto front is the set of models that no other model beats.",
)
body += section(
    "pareto-front",
    "",
    "The models nothing beats: the Pareto front",
    s1b,
    "A model is on the Pareto front when no other model is at least as good on both and better on one. Best on score alone is one end; which point is best depends on the speed you need.",
)
body += section(
    "pareto-fairness",
    "",
    "Admissions: can a causal fairness rule cost both goals?",
    s2,
    "The thick part of the front beats the rule&#8217;s policy on both goals.",
    "Schematic after Nilforoshan, Gaebler, Shroff &amp; Goel (2022), Causal conceptions of fairness and their consequences, ICML · not their numbers.",
)
if __name__ == "__main__":
    src = Path("lesson2-gates-s-preview.html").read_text().split("\n")
    head = "\n".join(src[:194]).replace(
        "Lesson 2 · Skips, gates and S beats", "Lesson 3 · Weighing tradeoffs"
    )
    tail = "\n".join(src[289:])
    Path("lesson3-tradeoffs.html").write_text(head + "\n" + body + tail)
    print(front)
