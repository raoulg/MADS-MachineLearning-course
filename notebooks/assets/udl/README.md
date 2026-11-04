# Figures from *Understanding Deep Learning*

Simon J.D. Prince, *Understanding Deep Learning*, MIT Press, 2023.
Book and figures: <https://udlbook.github.io/udlbook/> · repository: <https://github.com/udlbook/udlbook>

| File | Book figure | Page (PDF release v5.0.3) | Source file |
|---|---|---|---|
| `udl_fig3.3_shallow_buildup.pdf` / `.png` | Figure 3.3, "Computation for function in figure 3.2a" | p. 28 | `PDFFigures/UDLChap3PDF.zip` → `ShallowBuildUp.pdf` |
| `udl_fig3.5_shallow_approximate.pdf` / `.png` | Figure 3.5, "Approximation of a 1D function (dashed line) by a piecewise linear model" | p. 30 | `PDFFigures/UDLChap3PDF.zip` → `ShallowApproximate.pdf` |

The PDFs are the author's files, unchanged. The PNGs are the same PDFs rendered at 200 dpi
(`pdftoppm -r 200`), a format conversion only: no cropping, recolouring or other edits.

## Licence

The figures are licensed **CC BY-NC-ND 4.0**
(<https://creativecommons.org/licenses/by-nc-nd/4.0/>); `LICENSE.txt` is the licence note
shipped inside the figure archive. In practice:

- **BY** — always credit them, e.g. *"Figure 3.3 from S.J.D. Prince, Understanding Deep
  Learning (MIT Press, 2023), udlbook.github.io, CC BY-NC-ND 4.0."*
- **NC** — non-commercial use only. Teaching in a degree programme is fine; reusing them in
  paid training or commercial material is not, without permission from MIT Press.
- **ND** — share them unmodified. Do not crop, annotate, recolour or combine them into a new
  image. Showing a figure whole on a slide, with the credit next to it, is fine. To show a
  variation, redraw it from scratch (as `02_nonlinearity.ipynb` does with matplotlib).

The book's Python notebooks (`Notebooks/` in the same repository) are under the **MIT
licence** (© 2023 Simon Prince), so their code and parameter values can be reused freely
with that copyright notice.
