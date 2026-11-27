from pathlib import Path

"""The two Pareto slides of make_l3deck.py read the way exercise 05 reads its front:
accuracy against FLOPs per image, lower cost is better, so the front lies up and to the left.
Writes lesson3-tradeoffs-flops.html, for Raoul to set beside lesson3-tradeoffs.html."""

import math

from make_l3deck import (
    BG,
    VT,
    X0,
    X1,
    Y1,
    P,
    V,
    axes,
    dot,
    section,
    txt,
)
from make_l3deck import (
    pts as SPEED_PTS,
)

# the same cloud, mirrored: what was fast and good is now cheap and good
pts = [(X0 + X1 - x, y) for x, y in SPEED_PTS]


def dominated(p):
    return any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in pts)


front = sorted(p for p in pts if not dominated(p))  # from cheap (left) to dear
A = (X0 + X1 - 700, 350)
B = (X0 + X1 - 880, 200)
assert dominated(A) and B in pts


def box_left(x, y, fill=True):
    f = f'fill="{VT}" fill-opacity="0.55"' if fill else 'fill="none"'
    return f'<rect x="{X0}" y="{Y1}" width="{x - X0}" height="{y - Y1}" {f} stroke="{V}" stroke-width="2" stroke-dasharray="8 6"/>'


XL, YL = "FLOPs per image →", "accuracy ↑"
dots = "".join(dot(x, y) for x, y in pts)
s1 = axes(XL, YL) + (
    "<g data-build-in>"
    + box_left(A[0], A[1])
    + txt(X0 + 16, Y1 + 40, "better on both", 30, V, "start", 500)
    + f'<line x1="{A[0] - 12}" y1="{A[1] - 12}" x2="{B[0] + 14}" y2="{B[1] + 14}" stroke="{V}" stroke-width="4" marker-end="url(#av)"/>'
    + txt(B[0] - 26, B[1] - 22, "cheaper and higher", 30, V, "end", 500)
    + txt(A[0] + 24, A[1] + 50, "this model", 30, P, "start", 500)
    + "</g>"
)
s1 += dots + dot(*A, 16, BG, P, 5) + dot(*B, 16, VT, V, 4)

F = front[-3]
s1b = (
    axes(XL, YL)
    + dots
    + (
        "<g data-build-in>"
        + box_left(F[0], F[1], False)
        + dot(*F, 16, VT, V, 5)
        + txt(X0 + 16, Y1 + 40, "empty:", 30, V, "start", 500)
        + txt(X0 + 16, Y1 + 80, "nothing beats it on both", 30, V, "start", 500)
        + "</g>"
    )
)
fd = "M " + " L ".join(f"{x} {y}" for x, y in front)
s1b += (
    f'<g data-build-in><path d="{fd}" fill="none" stroke="{V}" stroke-width="5"/>'
    + "".join(dot(x, y, 15, V, V) for x, y in front)
    + txt(front[-5][0] - 24, front[-5][1] - 34, "Pareto front", 32, V, "end", 600)
    + "</g>"
)
top = min(front, key=lambda p: p[1])
a0 = (front[-3][0] + 24, front[-3][1] - 44)
a1 = (top[0] - 30, top[1] - 40)
mid = ((a0[0] + a1[0]) / 2, (a0[1] + a1[1]) / 2 - 16)
s1b += (
    "<g data-build-in>"
    + f'<circle cx="{top[0]}" cy="{top[1]}" r="26" fill="none" stroke="{P}" stroke-width="3"/>'
    + txt(top[0] + 36, top[1] - 30, "best on accuracy alone", 30, P, "start", 500)
    + f'<path d="M {a0[0]} {a0[1]} L {a1[0]} {a1[1]}" fill="none" stroke="{P}" stroke-width="3" stroke-dasharray="8 6" marker-end="url(#ap)"/>'
    + f'<g transform="rotate({math.degrees(math.atan2(a1[1] - a0[1], a1[0] - a0[0])):.1f} {mid[0]} {mid[1]})">'
    + txt(mid[0], mid[1], "more FLOPs, higher accuracy", 28, P, "middle", 500)
    + "</g></g>"
)

body = section(
    "pareto-beaten",
    "01 Pareto front",
    "Which model is best, if you care about two things?",
    s1,
    "This model is beaten: another model costs fewer FLOPs per image and is more accurate. The Pareto front is the set of models that no other model beats.",
)
body += section(
    "pareto-front",
    "",
    "The models nothing beats: the Pareto front",
    s1b,
    "A model is on the Pareto front when no other model is at least as good on both and better on one. Best on accuracy alone is one end; which point is best depends on the FLOPs you can spend.",
)
src = Path("lesson3-tradeoffs.html").read_text()
head = src[: src.index("<section")]
tail = src[src.rindex("</section>") + len("</section>\n") :]
Path("lesson3-tradeoffs-flops.html").write_text(head + "\n" + body + tail)
print(front)
