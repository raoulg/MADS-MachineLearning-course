# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "vectormesh",
#     "matplotlib",
#     "numpy",
#     "sentence-transformers",
#     "datasets",
#     "pillow",
# ]
# ///
"""Figures for the lesson 1 slides (presentations/lesson1-tensors.html), in the pttrn Paper palette.

Run from the repository root:

    uv run presentations/img/lesson1/make_figures.py

Writes PNGs next to this file and prints the numbers the slides quote.

Data:
- the published vector caches on the hub (pttrn-io/oxford-flowers-*, eurosat-*, dtd-*);
- Chelsea the cat from scikit-image (CC0, photo by Stefan van der Walt);
- six EuroSAT tiles from tanganke/eurosat (MIT), streamed;
- all-MiniLM-L6-v2 for the sentence vectors;
- the DTD head search of notebook 04, exported to data/dtd_search_trials.csv;
- the seed-spread table in planning/evidence.md section 3, typed in below.

Not made here: udl_fig3_3.png and udl_fig3_5.png are Figures 3.3 and 3.5 of Simon Prince,
Understanding Deep Learning (MIT Press, 2023), CC BY-NC-ND 4.0, rendered unchanged from
https://github.com/udlbook/udlbook/raw/main/PDFFigures/UDLChap3PDF.zip
(ShallowBuildUp.pdf and ShallowApproximate.pdf); their licence is udl_LICENSE.txt. The band_*.png files are the pttrn Rule 30
bands from the MADS-DAV decks.
"""

import csv
import glob
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import NullFormatter
from torch import nn

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]  # the course repository
DATA = OUT / "data"
DATA.mkdir(exist_ok=True)
REPO = OUT.parents[2]

BG = "#f4f1ec"
TEXT = "#111111"
ACCENT = "#1d5a73"
ACCENT_LIGHT = "#5e93a8"
SECOND = "#c43a14"
THIRD = "#8a6d3b"
GREY = "#2e2a26"
GREY_DIM = "#5a544d"
GREY_FAINT = "#cfc7bb"
INK_BG = "#141210"
INK_TEXT = "#f4f1ec"
INK_MUTED = "#b8b0a5"
CATEGORIES = [
    ACCENT,
    SECOND,
    THIRD,
    ACCENT_LIGHT,
    "#7a5a8c",
    "#6b7a3a",
    GREY,
    "#c9a66b",
    "#b77a6a",
    "#9c8f7d",
]
DIVERGING = LinearSegmentedColormap.from_list(
    "petrol_vermilion", [ACCENT, ACCENT_LIGHT, "#fdfcfa", "#e2481f", SECOND]
)

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": TEXT,
        "axes.labelcolor": GREY,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "axes.edgecolor": GREY_FAINT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "font.family": "sans-serif",
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "legend.fontsize": 14,
        "lines.linewidth": 2.2,
    }
)


def save(fig, name, facecolor=BG):
    fig.savefig(
        OUT / f"{name}.png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.15,
        facecolor=facecolor,
    )
    plt.close(fig)
    print(f"wrote {name}.png")


def cache(name: str, split: str):
    from vectormesh import VectorCache

    return VectorCache.from_hub(f"pttrn-io/{name}", split=split)


def xy(c):
    return c["embed"][:], c["label"][:]


# ---------------------------------------------------------------- the cat


def cat_image() -> np.ndarray:
    from PIL import Image

    # scikit-image's chelsea(), stored in the course assets (CC0); (H, W, 3) uint8, 300 x 451
    return np.asarray(
        Image.open(ROOT / "notebooks/assets/scikit-image/chelsea.png").convert("RGB")
    )


def channel_rgb(img: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(img)
    out[:, :, k] = img[:, :, k]
    return out


def cards(ax, images, dx, dy, w, h, edge=GREY_FAINT):
    """Draw images as a stack of cards, the first one in front."""
    n = len(images)
    for i in reversed(range(n)):
        x0, y0 = i * dx, i * dy
        ax.imshow(
            images[i],
            extent=(x0, x0 + w, y0, y0 + h),
            zorder=n - i,
            interpolation="lanczos",
        )
        ax.add_patch(
            Rectangle((x0, y0), w, h, fill=False, ec=edge, lw=1.2, zorder=n - i + 0.5)
        )


def fig_cover():
    img = cat_image()
    planes = [img] + [channel_rgb(img, k) for k in range(3)]
    fig, ax = plt.subplots(figsize=(7.5, 6.2), facecolor=INK_BG)
    ax.set_facecolor(INK_BG)
    h, w = 3.0, 4.51
    cards(ax, planes, 1.0, 0.75, w, h, edge="#5a5148")
    ax.set_xlim(-0.1, 3 * 1.0 + w + 0.1)
    ax.set_ylim(-0.1, 3 * 0.75 + h + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "cover_cat", facecolor=INK_BG)


def fig_cat_channels():
    img = cat_image()
    H, W, _ = img.shape
    fig = plt.figure(figsize=(16, 5.2))
    ax0 = fig.add_axes((0.0, 0.08, 0.30, 0.84))
    ax0.imshow(img)
    ax0.set_title("the photo", color=GREY)
    ax0.set_xlabel(f"(H, W, 3) = ({H}, {W}, 3)", color=GREY_DIM)
    ax0.set_xticks([])
    ax0.set_yticks([])
    names = ["R: red", "G: green", "B: blue"]
    for k in range(3):
        ax = fig.add_axes((0.335 + k * 0.225, 0.08, 0.21, 0.84))
        ax.imshow(channel_rgb(img, k))
        ax.set_title(f"channel {k} · {names[k]}", color=[SECOND, "#3d7a3d", ACCENT][k])
        ax.set_xlabel(f"({H}, {W})", color=GREY_DIM)
        ax.set_xticks([])
        ax.set_yticks([])
    save(fig, "cat_channels")
    t = torch.from_numpy(img).permute(2, 0, 1)
    print(
        f"cat: (H, W, 3) {img.shape} -> permute(2, 0, 1) -> (C, H, W) {tuple(t.shape)}"
    )
    print(f"cat: pixel at row 150, column 225 = {img[150, 225].tolist()}")


def eurosat_tiles():
    from datasets import load_dataset

    want = [
        "forest",
        "river",
        "highway or road",
        "residential buildings or homes or apartments",
        "lake or sea",
        "annual crop land",
    ]
    ds = load_dataset("tanganke/eurosat", split="test", streaming=True)
    assert ds.features is not None
    names = ds.features["label"].names
    found = {}
    for row in ds:
        name = names[row["label"]]
        if name in want and name not in found:
            found[name] = np.asarray(row["image"].convert("RGB"))
        if len(found) == len(want):
            break
    return [(n.split(" or ")[0].replace(" buildings", ""), found[n]) for n in want]


def fig_batch_images():
    tiles = eurosat_tiles()
    fig = plt.figure(figsize=(16, 4.6))
    for i, (name, im) in enumerate(tiles):
        ax = fig.add_axes((0.0 + i * 0.105, 0.38, 0.095, 0.40))
        ax.imshow(im, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{i}: {name}", fontsize=12, color=GREY)
        for s in ax.spines.values():
            s.set_visible(False)
    ax = fig.add_axes((0.66, 0.14, 0.34, 0.86))
    imgs = [im for _, im in tiles]
    cards(ax, imgs, 0.42, 0.32, 2.2, 2.2)
    ax.set_xlim(-0.1, 5 * 0.42 + 2.3)
    ax.set_ylim(-0.1, 5 * 0.32 + 2.3)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(
        0.0,
        0.22,
        "six tiles, each (C, H, W) = (3, 64, 64)",
        color=GREY_DIM,
        fontsize=16,
    )
    fig.text(
        0.66, 0.02, "stacked: (B, C, H, W) = (6, 3, 64, 64)", color=ACCENT, fontsize=16
    )
    save(fig, "batch_images")
    print(f"eurosat tiles: {[n for n, _ in tiles]}, each {imgs[0].shape}")


# ---------------------------------------------------------------- sentences


SENTENCE = "The cat sleeps on the warm windowsill."
DOCUMENTS = [
    "The cat sleeps on the warm windowsill.",
    "Train delays expected on the line to Utrecht.",
    "Add the flour slowly and keep stirring.",
    "The striker scored twice in the second half.",
    "Interest rates stay unchanged this quarter.",
]
ONE_DOCUMENT = [
    "Our cat Chelsea arrived from the shelter in March.",
    "For the first week she hid under the sofa.",
    "Then she discovered the windowsill above the radiator.",
    "Now she sleeps there every afternoon.",
    "At six she walks to the kitchen and waits for dinner.",
]


def fig_sentences():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    v = model.encode([SENTENCE], convert_to_tensor=True)[0].cpu()
    print(
        f"sentence vector: {tuple(v.shape)}, first five {[round(float(x), 3) for x in v[:5]]}"
    )
    lim = float(v.abs().max())

    fig, ax = plt.subplots(figsize=(16, 1.5))
    ax.imshow(
        v.unsqueeze(0).numpy(),
        cmap=DIVERGING,
        vmin=-lim,
        vmax=lim,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_yticks([])
    ax.set_xticks([0, 63, 127, 191, 255, 319, 383])
    ax.set_xticklabels(["0", "63", "127", "191", "255", "319", "383"])
    ax.set_xlabel("position in the vector (384 numbers)")
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "sentence_vector")

    batch = model.encode(DOCUMENTS, convert_to_tensor=True).cpu()
    chunks = model.encode(ONE_DOCUMENT, convert_to_tensor=True).cpu()
    D = 40
    lim = float(max(batch[:, :D].abs().max(), chunks[:, :D].abs().max()))
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.4), gridspec_kw={"wspace": 0.75})
    for ax, t, labels, title, col in [
        (
            axes[0],
            batch,
            DOCUMENTS,
            "(B, D) = (5, 384): five documents, stacked",
            ACCENT,
        ),
        (
            axes[1],
            chunks,
            ONE_DOCUMENT,
            "(C, D) = (5, 384): one document, five chunks",
            SECOND,
        ),
    ]:
        ax.imshow(
            t[:, :D].numpy(),
            cmap=DIVERGING,
            vmin=-lim,
            vmax=lim,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_yticks(range(5))
        ax.set_yticklabels(
            [s if len(s) < 34 else s[:32] + "…" for s in labels], fontsize=12
        )
        ax.set_xticks([0, 10, 20, 30, 39])
        ax.set_xlabel(f"first {D} of 384 numbers")
        ax.set_title(title, color=col, fontsize=15)
        for s in ax.spines.values():
            s.set_visible(False)
    save(fig, "batch_vs_chunk")
    sim_batch = torch.nn.functional.cosine_similarity(
        batch[:, None], batch[None], dim=-1
    )
    sim_chunk = torch.nn.functional.cosine_similarity(
        chunks[:, None], chunks[None], dim=-1
    )
    off = ~torch.eye(5, dtype=torch.bool)
    print(
        f"mean cosine between rows: batch {sim_batch[off].mean():.2f}, chunks {sim_chunk[off].mean():.2f}"
    )


# ---------------------------------------------------------------- the ladder


def grid(ax, x, y, rows, cols, colors, cell=0.32, alpha=1.0, z=1):
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(
                Rectangle(
                    (x + c * cell, y - (r + 1) * cell),
                    cell,
                    cell,
                    fc=colors[r % len(colors)],
                    ec=BG,
                    lw=1.5,
                    alpha=alpha,
                    zorder=z,
                )
            )


def arrow(ax, a, b, text, color=GREY_DIM, offset=(0, 0.25), ha="center"):
    ax.add_patch(
        FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=22, lw=2, color=color)
    )
    mx, my = (a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1]
    ax.text(mx, my, text, ha=ha, va="center", fontsize=15, color=color)


def fig_ladder():
    fig, ax = plt.subplots(figsize=(16, 9.4))
    cols = 8
    chunk_shades = ["#1d5a73", "#2f6f88", "#4a86a0", "#6a9db3"]
    items = [ACCENT, SECOND, THIRD, "#7a5a8c"]
    top, bottom = 8.6, 3.9
    # one item
    ax.text(0.5, top + 0.35, "(D)  one item", fontsize=18, color=TEXT)
    grid(ax, 0.5, top, 1, cols, [ACCENT])
    ax.text(
        0.5, top - 0.75, "a photo, a sentence: one vector", fontsize=14, color=GREY_DIM
    )
    ax.text(8.6, top + 0.35, "(C, D)  one chunked item", fontsize=18, color=TEXT)
    grid(ax, 8.6, top, 4, cols, chunk_shades)
    ax.text(
        8.6,
        top - 1.75,
        "one long document, one row per chunk",
        fontsize=14,
        color=GREY_DIM,
    )
    # a batch
    ax.text(0.5, bottom + 0.35, "(B, D)  a batch", fontsize=18, color=TEXT)
    grid(ax, 0.5, bottom, 4, cols, items)
    ax.text(
        0.5,
        bottom - 1.75,
        "four different items, one row each",
        fontsize=14,
        color=GREY_DIM,
    )
    ax.text(
        8.6,
        bottom + 0.35 + 0.8,
        "(B, C, D)  a batch of chunked items",
        fontsize=18,
        color=TEXT,
    )
    for b in reversed(range(3)):
        grid(
            ax,
            8.6 + b * 0.45,
            bottom + b * 0.35,
            4,
            cols,
            [items[b]] * 4,
            alpha=1.0 - 0.18 * b,
            z=5 - b,
        )
    ax.text(
        8.6,
        bottom - 1.75,
        "three documents, each four chunks",
        fontsize=14,
        color=GREY_DIM,
    )
    # moves between the rungs
    arrow(
        ax,
        (3.7, top - 0.15),
        (8.2, top - 0.15),
        "split before encoding",
        offset=(0, 0.3),
    )
    arrow(
        ax,
        (1.7, top - 1.1),
        (1.7, bottom + 0.75),
        "  stack B items",
        offset=(0, 0),
        ha="left",
    )
    arrow(
        ax,
        (9.9, top - 2.15),
        (9.9, bottom + 1.6),
        "  stack, after padding to the same C",
        offset=(0, 0),
        ha="left",
    )
    arrow(
        ax,
        (8.2, bottom - 0.9),
        (3.7, bottom - 0.9),
        "aggregate over C, e.g. the mean",
        color=SECOND,
        offset=(0, 0.3),
    )
    ax.set_xlim(0, 17)
    ax.set_ylim(1.7, 9.3)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "ladder")


# ---------------------------------------------------------------- the cache


def fig_flowers_pca():
    te = cache("oxford-flowers-dinov2-small", "test")
    tr = cache("oxford-flowers-dinov2-small", "train")
    Xte, yte = xy(te)
    Xtr, ytr = xy(tr)
    A = torch.nn.functional.normalize(Xtr, dim=1)
    B = torch.nn.functional.normalize(Xte, dim=1)
    acc = (ytr[(B @ A.T).argmax(1)] == yte).float().mean()
    print(f"flowers 1-NN, dinov2-small, no training: {acc:.4f}  (chance {1 / 102:.4f})")

    names = te.features["label"].names
    rng = torch.Generator().manual_seed(0)
    classes = torch.randperm(len(names), generator=rng)[:8]
    mask = torch.isin(yte, classes)
    U, S, _ = torch.pca_lowrank(Xte[mask], q=2)
    coords = (U * S).numpy()
    lab = yte[mask].numpy()
    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    for i, c in enumerate(classes.tolist()):
        p = coords[lab == c]
        ax.scatter(p[:, 0], p[:, 1], s=42, color=CATEGORIES[i], alpha=0.9, ec="none")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xticks([])
    ax.set_yticks([])
    save(fig, "flowers_pca")
    print(
        f"flowers pca: {mask.sum().item()} test photos of 8 species: {[names[c] for c in classes.tolist()]}"
    )


def fig_ml_types():
    """The same flower vectors twice: with their labels, and without."""
    te = cache("oxford-flowers-dinov2-small", "test")
    Xte, yte = xy(te)
    rng = torch.Generator().manual_seed(0)
    classes = torch.randperm(102, generator=rng)[:8]
    mask = torch.isin(yte, classes)
    U, S, _ = torch.pca_lowrank(Xte[mask], q=2)
    coords = (U * S).numpy()
    lab = yte[mask].numpy()
    for name, coloured in [("ml_supervised", True), ("ml_unsupervised", False)]:
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        for i, c in enumerate(classes.tolist()):
            p = coords[lab == c]
            ax.scatter(
                p[:, 0],
                p[:, 1],
                s=26,
                color=CATEGORIES[i] if coloured else GREY_DIM,
                alpha=0.85,
                ec="none",
            )
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GREY_FAINT)
        save(fig, name)


def head_seconds_eurosat(epochs: int = 10) -> tuple[float, float]:
    Xtr, ytr = xy(cache("eurosat-dinov2-small", "train"))
    Xte, yte = xy(cache("eurosat-dinov2-small", "test"))
    torch.manual_seed(0)
    model = nn.Linear(384, 10)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    start = time.time()
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 256):
            idx = perm[i : i + 256]
            opt.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
    secs = time.time() - start
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean()
    return secs, float(acc)


def fig_economics():
    head, acc = head_seconds_eurosat()
    rows = 8000
    rate = 81.0  # img/s, dinov2-small on an Apple-silicon CPU: planning/evidence.md section 4
    encode = rows / rate
    n = np.arange(1, 101)
    in_loop = n * (encode + head) / 60
    once = (encode + n * head) / 60
    print(
        f"eurosat: head of 10 epochs over 6000 rows {head:.2f}s (test acc {acc:.4f}); encoding 8000 rows at {rate:.0f} img/s {encode:.0f}s"
    )
    print(
        f"100 experiments: encoder in the loop {in_loop[-1]:.0f} min, encoded once {once[-1]:.1f} min"
    )
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ax.plot(n, in_loop, color=GREY_DIM, label="encoder in the loop")
    ax.plot(n, once, color=ACCENT, label="encode once, then train heads")
    ax.text(100, in_loop[-1], f"  {in_loop[-1]:.0f} min", va="center", color=GREY_DIM)
    ax.text(100, once[-1], f"  {once[-1]:.1f} min", va="center", color=ACCENT)
    ax.set_xlabel("heads trained (a search of 100 trials)")
    ax.set_ylabel("minutes")
    ax.set_xlim(0, 112)
    ax.legend(loc="upper left")
    save(fig, "economics")


def fig_encoders():
    # full published caches, two-line head: planning/evidence.md section 2
    table = {
        "dinov2-small": {"eurosat": 0.9502, "dtd": 0.8033, "flowers": 0.9932},
        "resnet-18": {"eurosat": 0.9380, "dtd": 0.6589, "flowers": 0.9025},
        "mobilenet_v2": {"eurosat": 0.9258, "flowers": 0.9082},
    }
    datasets_ = [
        ("eurosat", "EuroSAT\nsatellite, 10"),
        ("flowers", "Flowers\n102 species"),
        ("dtd", "DTD\ntextures, 47"),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    marks = {
        "dinov2-small": ("o", ACCENT),
        "resnet-18": ("s", SECOND),
        "mobilenet_v2": ("D", THIRD),
    }
    for j, (key, _) in enumerate(datasets_):
        vals = [table[e][key] for e in table if key in table[e]]
        ax.plot(
            [j, j],
            [min(vals), max(vals)],
            color=GREY_FAINT,
            lw=6,
            solid_capstyle="round",
            zorder=1,
        )
        ax.text(
            j + 0.13,
            (min(vals) + max(vals)) / 2,
            f"{100 * (max(vals) - min(vals)):.1f} pt",
            color=GREY_DIM,
            va="center",
            fontsize=14,
        )
        for e, (m, c) in marks.items():
            if key in table[e]:
                ax.scatter(
                    j,
                    table[e][key],
                    marker=m,
                    s=140,
                    color=c,
                    zorder=3,
                    label=e if j == 0 else None,
                )
    ax.set_xticks(range(3))
    ax.set_xticklabels([lab for _, lab in datasets_])
    ax.set_xlim(-0.5, 2.7)
    ax.set_ylabel("test accuracy, same head")
    ax.legend(loc="lower left")
    save(fig, "encoders")


# ---------------------------------------------------------------- nonlinearity


def fig_collapse():
    torch.manual_seed(0)
    width = 12
    layers = [nn.Linear(width, width, bias=False) for _ in range(5)]
    W = layers[-1].weight
    for layer in reversed(layers[:-1]):
        W = W @ layer.weight
    x = torch.rand(8, width)
    with torch.no_grad():
        diff = (nn.Sequential(*layers)(x) - x @ W.T).abs().max()
    print(
        f"collapse: five {width}x{width} layers vs one matrix, max difference {diff:.1e}"
    )
    fig, axes = plt.subplots(
        1, 7, figsize=(16, 2.9), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 0.35, 1]}
    )
    for i in range(5):
        w = layers[4 - i].weight.detach()
        lim = float(w.abs().max())
        axes[i].imshow(w, cmap=DIVERGING, vmin=-lim, vmax=lim)
        axes[i].set_title(f"W{5 - i}", color=GREY)
    axes[5].text(0.5, 0.5, "=", fontsize=40, ha="center", va="center", color=GREY_DIM)
    Wd = W.detach()
    lim = float(Wd.abs().max())
    axes[6].imshow(Wd, cmap=DIVERGING, vmin=-lim, vmax=lim)
    axes[6].set_title("W", color=SECOND)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[5].axis("off")
    save(fig, "collapse")


def fig_sine():
    xs = torch.linspace(-3, 3, 200).unsqueeze(1)
    true = torch.sin(3 * xs)
    torch.manual_seed(1)
    ys = true + 0.15 * torch.randn_like(true)

    def fit(model, steps=2000):
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(steps):
            opt.zero_grad()
            loss = ((model(xs) - ys) ** 2).mean()
            loss.backward()
            opt.step()
        return model, float(loss.detach())

    torch.manual_seed(0)
    lin, lin_loss = fit(
        nn.Sequential(nn.Linear(1, 64), nn.Linear(64, 64), nn.Linear(64, 1))
    )
    torch.manual_seed(0)
    gel, gel_loss = fit(
        nn.Sequential(
            nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1)
        )
    )
    print(f"sine: three linear layers MSE {lin_loss:.3f}; with GELU {gel_loss:.3f}")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    with torch.no_grad():
        ax.scatter(xs, ys, s=12, color=GREY_FAINT, label="noisy data")
        ax.plot(xs, true, color=GREY, lw=1.5, ls=":", label="sin(3x)")
        ax.plot(
            xs,
            lin(xs),
            color=GREY_DIM,
            lw=2.6,
            label=f"three Linear layers  (MSE {lin_loss:.2f})",
        )
        ax.plot(
            xs,
            gel(xs),
            color=SECOND,
            lw=2.6,
            label=f"the same, with GELU between  (MSE {gel_loss:.3f})",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower right", fontsize=13)
    ax.set_ylim(-1.9, 1.6)
    save(fig, "sine_fit")


def fig_activations():
    z = torch.linspace(-4, 4, 400, requires_grad=True)
    acts = {
        "ReLU": torch.relu,
        "Leaky ReLU": lambda t: torch.nn.functional.leaky_relu(t, 0.1),
        "ELU": torch.nn.functional.elu,
        "GELU": torch.nn.functional.gelu,
        "sigmoid": torch.sigmoid,
    }
    colors = [ACCENT, ACCENT_LIGHT, THIRD, SECOND, GREY_DIM]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for (name, fn), c in zip(acts.items(), colors):
        y = fn(z)
        (g,) = torch.autograd.grad(y.sum(), z)
        axes[0].plot(z.detach(), y.detach(), color=c, label=name)
        axes[1].plot(z.detach(), g, color=c, label=name)
    axes[0].set_title("the activation σ(z)", color=GREY)
    axes[1].set_title("its slope: what the gradient is multiplied by", color=GREY)
    for ax in axes:
        ax.axhline(0, color=GREY_FAINT, lw=1)
        ax.axvline(0, color=GREY_FAINT, lw=1)
        ax.set_xlabel("z")
    axes[0].set_ylim(-1.5, 4)
    axes[1].set_ylim(-0.15, 1.25)
    axes[0].legend(loc="upper left", fontsize=13)
    save(fig, "activations")


# ---------------------------------------------------------------- training


def fig_batching():
    tr = cache("eurosat-dinov2-small", "train")
    names = tr.features["label"].names
    rows = [tr[i] for i in range(4)]
    X = torch.stack([r["embed"] for r in rows])
    y = torch.stack([r["label"] for r in rows])
    D = 24
    lim = float(X[:, :D].abs().max())
    fig = plt.figure(figsize=(16, 4.4))
    for i in range(4):
        ax = fig.add_axes((0.0, 0.80 - i * 0.22, 0.32, 0.12))
        ax.imshow(X[i : i + 1, :D], cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"row {i}: (384,)   label {int(y[i])} = {names[int(y[i])]}",
            fontsize=12,
            color=GREY_DIM,
            loc="left",
        )
        for s in ax.spines.values():
            s.set_visible(False)
    fig.text(0.36, 0.48, "torch.stack  →", fontsize=20, color=ACCENT, va="center")
    ax = fig.add_axes((0.52, 0.12, 0.30, 0.72))
    ax.imshow(X[:, :D], cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(4))
    ax.set_xticks([])
    ax.set_ylabel("B")
    ax.set_xlabel(f"D (first {D} of 384)")
    ax.set_title("X: (B, D) = (4, 384)", color=ACCENT)
    for s in ax.spines.values():
        s.set_visible(False)
    ax2 = fig.add_axes((0.86, 0.12, 0.05, 0.72))
    ax2.imshow(
        y.unsqueeze(1).numpy(),
        cmap=LinearSegmentedColormap.from_list("g", [BG, BG]),
        aspect="auto",
    )
    for i in range(4):
        ax2.text(
            0, i, str(int(y[i])), ha="center", va="center", fontsize=18, color=SECOND
        )
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("y: (B,)", color=SECOND)
    for s in ax2.spines.values():
        s.set_color(GREY_FAINT)
    save(fig, "batching")


def fig_loss_metric():
    p = np.linspace(0.01, 0.99, 400)
    ce = -np.log(p)
    acc = (p > 0.5).astype(float)
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.plot(p, ce, color=ACCENT, label="loss: cross-entropy, −log p")
    ax.step(
        p, acc, where="mid", color=SECOND, label="metric: accuracy (right or wrong)"
    )
    ax.set_xlabel("probability the model gives the true class (two classes)")
    ax.set_ylabel("value")
    ax.set_ylim(-0.1, 3.2)
    ax.legend(loc="upper right")
    ax.annotate(
        "flat: no direction to move in",
        xy=(0.8, 1.0),
        xytext=(0.6, 1.6),
        color=SECOND,
        arrowprops={"arrowstyle": "-|>", "color": SECOND},
        fontsize=14,
    )
    save(fig, "loss_metric")


def fig_optimizer():
    a, b = 1.0, 12.0  # a long, narrow valley with two weights

    def loss(w):
        return 0.5 * (a * w[0] ** 2 + b * w[1] ** 2)

    def grad(w):
        return np.array([a * w[0], b * w[1]])

    start = np.array([-4.5, 1.6])

    def sgd(lr, steps=40):
        w, path = start.copy(), [start.copy()]
        for _ in range(steps):
            w = w - lr * grad(w)
            path.append(w.copy())
        return np.array(path)

    def adam(lr=0.15, steps=40, b1=0.9, b2=0.999, eps=1e-8):
        w, m, v, path = start.copy(), np.zeros(2), np.zeros(2), [start.copy()]
        for t in range(1, steps + 1):
            g = grad(w)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            w = w - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
            path.append(w.copy())
        return np.array(path)

    gx, gy = np.meshgrid(np.linspace(-5, 5, 300), np.linspace(-2.2, 2.2, 300))
    Z = loss(np.array([gx, gy]))
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.contour(
        gx, gy, Z, levels=np.geomspace(0.05, 40, 14), colors=GREY_FAINT, linewidths=1
    )
    runs = [
        (sgd(0.02), GREY_DIM, "SGD, small step: slow"),
        (sgd(0.16), SECOND, "SGD, large step: zigzags"),
        (adam(), ACCENT, "Adam: a step size per weight"),
    ]
    for path, c, lab in runs:
        ax.plot(path[:, 0], path[:, 1], "-o", color=c, ms=4, lw=1.8, label=lab)
    ax.scatter([0], [0], marker="*", s=300, color=TEXT, zorder=5)
    ax.scatter([start[0]], [start[1]], s=80, color=TEXT, zorder=5)
    ax.text(start[0], start[1] + 0.25, "start", ha="center", fontsize=14)
    ax.set_xlabel("weight 1")
    ax.set_ylabel("weight 2")
    ax.legend(loc="lower right", fontsize=13)
    ax.set_aspect("equal")
    save(fig, "optimizer")


def fig_training_curve():
    Xtr, ytr = xy(cache("eurosat-dinov2-small", "train"))
    Xte, yte = xy(cache("eurosat-dinov2-small", "test"))
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(384, 384), nn.GELU(), nn.Linear(384, 10))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    hist = []
    for epoch in range(20):
        perm = torch.randperm(len(Xtr))
        total, n = 0.0, 0
        for i in range(0, len(Xtr), 64):
            idx = perm[i : i + 64]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            n += len(idx)
        with torch.no_grad():
            test_loss = float(loss_fn(model(Xte), yte))
            acc = float((model(Xte).argmax(1) == yte).float().mean())
        hist.append((total / n, test_loss, acc))
    h = np.array(hist)
    print(
        f"eurosat training curve: final train loss {h[-1, 0]:.3f}, test loss {h[-1, 1]:.3f}, test acc {h[-1, 2]:.4f}; best test loss at epoch {int(h[:, 1].argmin()) + 1}"
    )
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ep = np.arange(1, 21)
    ax.plot(ep, h[:, 0], color=ACCENT, label="train loss")
    ax.plot(ep, h[:, 1], color=ACCENT_LIGHT, ls="--", label="test loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax2 = ax.twinx()
    ax2.plot(ep, h[:, 2], color=SECOND, label="test accuracy")
    ax2.set_ylabel("test accuracy", color=SECOND)
    ax2.spines["right"].set_visible(True)
    ax2.tick_params(axis="y", colors=SECOND)
    ax2.set_ylim(0.90, 0.97)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(
        lines,
        [str(line.get_label()) for line in lines],
        loc="center right",
        bbox_to_anchor=(1.0, 0.35),
    )
    save(fig, "training_curve")


# ---------------------------------------------------------------- search


ENC_ORDER = ["resnet-18", "dinov2-small", "dinov2-base", "dinov2-large"]
ENC_COLOR = {
    "resnet-18": GREY_FAINT,
    "dinov2-small": ACCENT_LIGHT,
    "dinov2-base": ACCENT,
    "dinov2-large": "#0d2c38",
}
N_TRIALS = 24  # notebook 04's search budget


def export_trials() -> Path:
    """The trials of the most recent run of notebook 04's search, as a csv."""
    target = DATA / "dtd_search_trials.csv"
    src = REPO / "notebooks/1_pytorch_intro/ray_results/dtd_head_search"
    rows = []
    for d in glob.glob(str(src / "trainable_*")):
        try:
            r = json.loads(Path(d, "result.json").read_text().strip().splitlines()[-1])
        except (OSError, ValueError, IndexError):
            continue
        c = r["config"]
        if "n_hidden_layers" in c and c["epochs"] == 15:
            rows.append(
                {
                    "date": r["date"],
                    **{
                        k: c[k]
                        for k in (
                            "encoder",
                            "n_hidden_layers",
                            "hidden_dim",
                            "sigma",
                            "lr",
                        )
                    },
                    "accuracy": r["Accuracy"],
                }
            )
    if rows:
        rows = sorted(rows, key=lambda r: r["date"])[-N_TRIALS:]
        with target.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return target


# DTD, dinov2-small, 5 seeds, 15 epochs: planning/evidence.md section 3, "What 'baseline' means"
DTD_KNN = 0.770
DTD_LINEAR = (0.789, 0.003)  # nn.Linear(384, 47), notebook 04's baseline


def fig_search():
    path = export_trials()
    with path.open() as f:
        rows = list(csv.DictReader(f))
    base, std = DTD_LINEAR
    acc = np.array([float(r["accuracy"]) for r in rows])
    best = rows[int(acc.argmax())]
    print(
        f"search: {len(rows)} trials; best {acc.max():.4f} with {best['encoder']}, n_hidden_layers {best['n_hidden_layers']}"
    )
    for e in ENC_ORDER:
        a = [float(r["accuracy"]) for r in rows if r["encoder"] == e]
        if a:
            print(f"   {e:<13} {len(a):>2} trials  {min(a):.4f} - {max(a):.4f}")
    fig, axes = plt.subplots(
        1, 4, figsize=(17, 4.6), sharey=True, gridspec_kw={"wspace": 0.08}
    )
    rng = np.random.default_rng(0)
    panels = [
        ("encoder", ENC_ORDER, lambda r: r["encoder"]),
        ("n_hidden_layers", ["0", "1", "2"], lambda r: r["n_hidden_layers"]),
    ]
    for ax, (col, cats, get) in zip(axes[:2], panels):
        for j, cat in enumerate(cats):
            sub = [r for r in rows if get(r) == cat]
            a = [float(r["accuracy"]) for r in sub]
            if a:
                ax.scatter(
                    j + rng.uniform(-0.12, 0.12, len(a)),
                    a,
                    s=56,
                    c=[ENC_COLOR[r["encoder"]] for r in sub],
                    ec="none",
                )
                ax.plot([j - 0.25, j + 0.25], [np.mean(a)] * 2, color=SECOND, lw=3)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=20 if col == "encoder" else 0, fontsize=13)
        ax.set_xlim(-0.6, len(cats) - 0.4)
        ax.set_title(col, color=GREY)
    for ax, col, log in [(axes[2], "sigma", False), (axes[3], "lr", True)]:
        ax.scatter(
            [float(r[col]) for r in rows],
            acc,
            s=56,
            c=[ENC_COLOR[r["encoder"]] for r in rows],
            ec="none",
        )
        if log:
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(col, color=GREY)
    for ax in axes:
        ax.axhspan(base - std, base + std, color=GREY_FAINT, alpha=0.6, lw=0)
        ax.axhline(base, color=GREY, ls="--", lw=1.2)
    axes[0].set_ylabel("test accuracy")
    save(fig, "search_axes")


def fig_fusion():
    # DTD, dinov2-small (+ resnet-18 for fusion), 5 seeds, 15 epochs: planning/evidence.md section 3
    rows = [
        ("1-NN\nno training", DTD_KNN, None, GREY_DIM),
        ("nn.Linear\nbaseline", *DTD_LINEAR, ACCENT),
        ("one hidden\nlayer", 0.812, None, SECOND),
        ("fusion 896\n→ nn.Linear", 0.810, 0.003, SECOND),
        ("fusion 896\n→ one hidden", 0.824, None, SECOND),
    ]
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    for i, (lab, m, s, c) in enumerate(rows):
        ax.errorbar(i, m, yerr=s, fmt="o", ms=11, color=c, capsize=8, lw=2.4)
        ax.text(
            i + 0.14,
            m,
            f"{m:.3f}" + (f" ± {s:.3f}" if s else ""),
            va="center",
            fontsize=13,
            color=c,
        )
    base, std = DTD_LINEAR
    ax.axhspan(base - std, base + std, color=GREY_FAINT, alpha=0.6, lw=0)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=13)
    ax.set_xlim(-0.4, len(rows) - 0.1)
    ax.set_ylim(0.76, 0.835)
    ax.set_ylabel("test accuracy (± std over 5 seeds)")
    save(fig, "fusion_seeds")


FIGURES = [
    fig_cover,
    fig_cat_channels,
    fig_batch_images,
    fig_sentences,
    fig_ladder,
    fig_flowers_pca,
    fig_ml_types,
    fig_economics,
    fig_encoders,
    fig_collapse,
    fig_sine,
    fig_activations,
    fig_batching,
    fig_loss_metric,
    fig_optimizer,
    fig_training_curve,
    fig_search,
    fig_fusion,
]

if __name__ == "__main__":
    import sys

    # with arguments, only the named figures: make_figures.py fig_sine fig_search
    for f in FIGURES:
        if len(sys.argv) == 1 or f.__name__ in sys.argv[1:]:
            f()
