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
- two photos of each of oxford-flowers (nkirschi/oxford-flowers), EuroSAT and DTD (tanganke/dtd), streamed;
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


def fig_samples():
    """Two photos of each dataset of the comparison table, as one strip per dataset."""
    from datasets import load_dataset
    from PIL import Image

    side, gap = 300, 16

    def first_two(repo: str, want: list[str] | None):
        ds = load_dataset(repo, split="test", streaming=True)
        assert ds.features is not None
        names = ds.features["label"].names
        found: dict[str, Image.Image] = {}
        for row in ds:
            name = names[row["label"]]
            if (want is None or name in want) and name not in found:
                found[name] = row["image"].convert("RGB")
            if len(found) == 2:
                break
        return list(found.items())

    sets = {
        "samples_flowers": first_two("nkirschi/oxford-flowers", None),
        "samples_eurosat": first_two(
            "tanganke/eurosat",
            ["annual crop land", "residential buildings or homes or apartments"],
        ),
        "samples_dtd": first_two("tanganke/dtd", ["braided", "cracked"]),
    }
    for stem, pair in sets.items():
        strip = Image.new("RGB", (2 * side + gap, side), BG)
        for k, (_, im) in enumerate(pair):
            w, h = im.size
            s = min(w, h)
            im = im.crop(
                ((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)
            )
            strip.paste(
                im.resize((side, side), Image.Resampling.BICUBIC), (k * (side + gap), 0)
            )
        strip.save(OUT / f"{stem}.png")
        print(f"wrote {stem}.png: {[n for n, _ in pair]}")


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


def depth(ax, start, n, step, text, color):
    """A bracket along a stack's depth axis: n card slots from `start`, labelled `text`."""
    (x0, y0), (dx, dy) = start, step
    x1, y1 = x0 + n * dx, y0 + n * dy
    ax.plot([x0, x1], [y0, y1], color=color, lw=3, solid_capstyle="butt", zorder=50)
    for x, y in ((x0, y0), (x1, y1)):
        ax.plot(
            [x - 0.09, x + 0.09], [y + 0.09, y - 0.09], color=color, lw=2.5, zorder=50
        )
    ax.text(
        (x0 + x1) / 2 + 0.25,
        (y0 + y1) / 2 - 0.05,
        text,
        color=color,
        fontsize=20,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=51,
    )


def fig_unit_axis():
    img = cat_image()
    gray = img.astype(float).mean(axis=-1)
    H, W = gray.shape
    fig = plt.figure(figsize=(16, 7.2))

    # top row: the cat, grey and in colour; channels first, as on the slides
    w, h = 3.6, 3.6 * H / W
    dx, dy = 0.42, 0.34
    for k, (planes, n, label, color) in enumerate(
        [
            (None, 0, f"(H, W) = ({H}, {W})", GREY),
            ([gray], 1, f"(1, H, W) = (1, {H}, {W})", SECOND),
            (
                [channel_rgb(img, c) for c in range(3)],
                3,
                f"(3, H, W) = (3, {H}, {W})",
                GREY,
            ),
        ]
    ):
        ax = fig.add_axes((0.0 + k * 0.335, 0.47, 0.31, 0.50))
        if planes is None:
            ax.imshow(gray, cmap="gray", extent=(0, w, 0, h), interpolation="lanczos")
        else:
            cards(
                ax,
                [
                    np.stack([p] * 3, -1).astype(np.uint8) if p.ndim == 2 else p
                    for p in planes
                ],
                dx,
                dy,
                w,
                h,
            )
        if n:
            depth(ax, (w + 0.08, 0.0), n, (dx, dy), str(n), color)
        ax.set_xlim(-0.05, w + 3 * dx + 0.6)
        ax.set_ylim(-0.05, h + 3 * dy + 0.05)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(label, color=color, fontsize=22)

    # bottom row: the ladder's shapes, drawn as on the ladder slide
    chunk_shades = ["#1d5a73", "#2f6f88", "#4a86a0", "#6a9db3"]
    items = [
        ACCENT,
        "#6b7a3a",
        THIRD,
        "#7a5a8c",
    ]  # no vermilion: here it marks the axis of length 1
    cols, cell = 8, 0.32
    gw = cols * cell
    sdx, sdy = 0.3, 0.42
    panels = [
        ("(C, D)", GREY, "one document"),
        ("(1, C, D)", SECOND, "a batch of one document"),
        ("(B, D)", GREY, "four documents"),
        ("(B, 1, D)", SECOND, "four documents, one chunk each"),
    ]
    for k, (label, color, sub) in enumerate(panels):
        ax = fig.add_axes((0.0 + k * 0.25, 0.0, 0.24, 0.40))
        if k == 0:
            grid(ax, 0, 4 * cell, 4, cols, chunk_shades)
        elif k == 1:
            grid(ax, 0, 4 * cell, 4, cols, chunk_shades)
            depth(ax, (gw + 0.06, 0.0), 1, (sdx, sdy), "B = 1", SECOND)
        elif k == 2:
            grid(ax, 0, 4 * cell, 4, cols, items)
        else:
            for b in reversed(range(4)):
                grid(ax, b * sdx, cell + b * sdy, 1, cols, [items[b]], z=5 - b)
            ax.plot([-0.1, -0.1], [0, cell], color=SECOND, lw=3)
            ax.text(
                -0.18,
                cell / 2,
                "C = 1",
                color=SECOND,
                fontsize=20,
                fontweight="bold",
                ha="right",
                va="center",
            )
            depth(ax, (gw + 0.06, 0.0), 4, (sdx, sdy), "B = 4", GREY)
        ax.set_xlim(-1.3, gw + 4 * sdx + 1.2)
        ax.set_ylim(-0.6, 4 * cell + 4 * sdy * 0.5 + 0.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(label, color=color, fontsize=22)
        ax.text(0, -0.3, sub, color=GREY_DIM, fontsize=18, va="top")
    save(fig, "unit_axis")
    g1 = torch.from_numpy(gray).unsqueeze(0)
    print(
        f"grey cat: {gray.shape} -> unsqueeze(0) -> {tuple(g1.shape)}, pixel [150, 225] = {gray[150, 225]:.1f}"
    )


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
    src = REPO / "notebooks/1_representation_and_head/ray_results/dtd_head_search"
    rows = []
    for d in glob.glob(str(src / "trainable_*")):
        try:
            r = json.loads(
                Path(d, "result.json")
                .read_text(encoding="utf-8")
                .strip()
                .splitlines()[-1]
            )
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
        with target.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return target


# DTD, dinov2-small, 5 seeds, 15 epochs: planning/evidence.md section 3, "What 'baseline' means"
DTD_KNN = 0.770
DTD_LINEAR = (0.789, 0.003)  # nn.Linear(384, 47), notebook 04's baseline


def fig_search():
    path = export_trials()
    with path.open(encoding="utf-8") as f:
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


# ---------------------------------------------------------------- context


def fig_doublings():
    """How many doublings are left before each candidate limit bends the curve.

    The colours are the pond's: petrol is logistic growth that stops at the limit, the dashed
    vermilion line is the exponential that keeps doubling.
    """
    limits = [
        ("intelligence: nerve → light", 3e6, ACCENT_LIGHT),
        ("material: silicon → sand", 1e12, ACCENT),
        ("energy: our power → the sun", 1e13, "#0d2c38"),
    ]
    x = np.linspace(0, 62, 700)
    fig, ax = plt.subplots(figsize=(13, 5.6))
    places = [(-1.0, 0.72, "right"), (-1.0, 0.45, "right"), (1.6, 0.30, "left")]
    for (name, ratio, c), (dx, yy, ha) in zip(limits, places):
        n = np.log2(ratio)  # doublings until today's amount reaches the limit
        share = 1 / (1 + (2.0**n - 1) * 2.0 ** (-x))  # logistic: today at 1 / ratio
        ax.plot(x, share, color=c, lw=3)
        ax.text(
            n + dx,
            yy,
            f"{name}\n{n:.0f} doublings",
            color=c,
            fontsize=14,
            va="center",
            ha=ha,
        )
    n0 = np.log2(limits[0][1])
    expo = 2.0 ** (x - n0)
    keep = expo < 1.1
    ax.plot(x[keep], expo[keep], color=SECOND, lw=2.6, ls="--")
    ax.text(
        n0 - 3.2,
        1.0,
        "exponential:\ndoubling forever",
        color=SECOND,
        fontsize=14,
        ha="right",
        va="center",
    )
    ax.scatter([0], [0], s=90, color=TEXT, zorder=5)
    ax.annotate(
        "today",
        xy=(0, 0.0),
        xytext=(1.2, 0.2),
        fontsize=15,
        arrowprops={"arrowstyle": "-|>", "color": TEXT},
    )
    ax.set_xlim(0, 62)
    ax.set_ylim(-0.03, 1.12)
    ax.set_yticks([0, 0.5, 1], ["0", "half", "the limit"])
    ax.set_xlabel("doublings from today")
    ax.set_ylabel("share of the limit")
    save(fig, "doublings")


# ---------------------------------------------------------------- the cache, again

FLOWER_NAMES = {  # Oxford 102 category numbers (the cache's label names) to species
    "90": "canna lily",
    "63": "black-eyed susan",
    "39": "siam tulip",
    "72": "azalea",
    "24": "red ginger",
    "66": "osteospermum",
    "73": "water lily",
    "100": "blanket flower",
}


def flower_photo(label_name: str, side: int = 200):
    from datasets import load_dataset
    from PIL import Image

    ds = load_dataset("nkirschi/oxford-flowers", split="test")
    names = ds.features["label"].names
    target = names.index(label_name)
    i = ds["label"].index(target)
    im = ds[i]["image"].convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(im.resize((side, side), Image.Resampling.BICUBIC))


def fig_flowers_pca_thumbs():
    """The eight species on two PCA axes, with one photo of each next to its group."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    te = cache("oxford-flowers-dinov2-small", "test")
    Xte, yte = xy(te)
    names = te.features["label"].names
    rng = torch.Generator().manual_seed(0)
    classes = torch.randperm(len(names), generator=rng)[:8].tolist()
    mask = torch.isin(yte, torch.tensor(classes))
    X = Xte[mask] - Xte[mask].mean(0)
    U, S, _ = torch.linalg.svd(
        X, full_matrices=False
    )  # exact, so the picture is stable
    coords = (U[:, :2] * S[:2]).numpy()
    lab = yte[mask].numpy()
    fig, ax = plt.subplots(figsize=(13, 7.6))
    cents = {}
    for i, c in enumerate(classes):
        p = coords[lab == c]
        ax.scatter(p[:, 0], p[:, 1], s=46, color=CATEGORIES[i], alpha=0.9, ec="none")
        cents[c] = (p.mean(0), CATEGORIES[i])
    # one photo per species, placed by hand around the three groups the plot shows
    place = {
        "66": (0.07, 0.86),
        "63": (0.07, 0.52),
        "100": (0.07, 0.18),
        "39": (0.56, 0.90),
        "24": (0.73, 0.90),
        "73": (0.90, 0.90),
        "90": (0.50, 0.17),
        "72": (0.68, 0.17),
    }
    for c, (m, col) in cents.items():
        pos = place[names[c]]
        img = OffsetImage(flower_photo(names[c], 160), zoom=0.40)
        ab = AnnotationBbox(
            img,
            m,
            xybox=pos,
            xycoords="data",
            boxcoords="axes fraction",
            frameon=True,
            pad=0.15,
            bboxprops={"edgecolor": col, "lw": 3},
            arrowprops={"arrowstyle": "-", "color": col, "lw": 1.5},
        )
        ax.add_artist(ab)
        ax.text(
            pos[0],
            pos[1] - 0.105,
            FLOWER_NAMES[names[c]],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
            color=col,
        )
        print(f"flowers pca: {FLOWER_NAMES[names[c]]:<16} centre {m.round(1)}")
    span = coords.max(0) - coords.min(0)
    ax.set_xlim(
        coords[:, 0].min() - 0.45 * span[0], coords[:, 0].max() + 0.15 * span[0]
    )
    ax.set_ylim(
        coords[:, 1].min() - 0.55 * span[1], coords[:, 1].max() + 0.55 * span[1]
    )
    ax.set_xlabel("first PCA direction")
    ax.set_ylabel("second PCA direction")
    ax.set_xticks([])
    ax.set_yticks([])
    save(fig, "flowers_pca")


def fig_encoders_thumbs():
    """The encoder comparison with one sample image of each dataset under its column."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from PIL import Image

    table = {
        "dinov2-small": {"eurosat": 0.9502, "dtd": 0.8033, "flowers": 0.9932},
        "resnet-18": {"eurosat": 0.9380, "dtd": 0.6589, "flowers": 0.9025},
        "mobilenet_v2": {"eurosat": 0.9258, "flowers": 0.9082},
    }
    datasets_ = [
        ("eurosat", "satellite tiles\n10 classes", "samples_eurosat"),
        ("flowers", "flowers\n102 species", "samples_flowers"),
        ("dtd", "textures\n47 kinds", "samples_dtd"),
    ]
    marks = {
        "dinov2-small": ("o", ACCENT),
        "resnet-18": ("s", SECOND),
        "mobilenet_v2": ("D", THIRD),
    }
    fig, ax = plt.subplots(figsize=(11, 7.2))
    for j, (key, lab, stem) in enumerate(datasets_):
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
            f"{100 * (max(vals) - min(vals)):.1f} points",
            color=GREY_DIM,
            va="center",
            fontsize=15,
        )
        for e, (m, c) in marks.items():
            if key in table[e]:
                ax.scatter(
                    j,
                    table[e][key],
                    marker=m,
                    s=160,
                    color=c,
                    zorder=3,
                    label=e if j == 0 else None,
                )
        if key == "eurosat":  # a tile whose landscape you recognise: a highway
            one = np.asarray(
                Image.fromarray(dict(eurosat_tiles())["highway"]).resize(
                    (300, 300), Image.Resampling.LANCZOS
                )
            )
        else:
            strip = np.asarray(Image.open(OUT / f"{stem}.png"))
            one = strip[:, : strip.shape[0]]  # the first of the two samples
        ab = AnnotationBbox(
            OffsetImage(one, zoom=0.36),
            (j, 0),
            xybox=(0, -62),
            xycoords=("data", "axes fraction"),
            boxcoords="offset points",
            frameon=False,
        )
        ax.add_artist(ab)
        ax.text(
            j,
            -0.50,
            lab,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=15,
        )
    ax.set_xticks([])
    ax.set_xlim(-0.5, 2.8)
    ax.set_ylabel("accuracy, same head")
    ax.legend(loc="lower left")
    fig.subplots_adjust(bottom=0.40)
    save(fig, "encoders")


# ---------------------------------------------------------------- tensors


def fig_audio_image():
    """A sound as a spectrogram (F, T), and the same numbers as a one-channel image (1, F, T)."""
    sr = 8000
    t = np.arange(int(1.2 * sr)) / sr
    wave = np.sin(2 * np.pi * (300 + 900 * t) * t) * np.exp(-1.5 * t) + 0.5 * np.sin(
        2 * np.pi * 1800 * t
    ) * (t > 0.6)
    n, hop = 256, 64
    frames = np.stack(
        [wave[i : i + n] * np.hanning(n) for i in range(0, len(wave) - n, hop)], axis=1
    )
    spec = np.log1p(np.abs(np.fft.rfft(frames, axis=0)))[:96]
    fig = plt.figure(figsize=(16, 4.2))
    a0 = fig.add_axes((0.0, 0.18, 0.27, 0.62))
    a0.plot(t, wave, color=GREY_DIM, lw=0.6)
    a0.set_title("a sound: (L,) samples", color=GREY, loc="left")
    a0.set_xticks([])
    a0.set_yticks([])
    a1 = fig.add_axes((0.33, 0.10, 0.25, 0.70))
    a1.imshow(spec, origin="lower", aspect="auto", cmap="cividis")
    a1.set_title("its spectrogram: (F, T)", color=GREY, loc="left")
    a1.set_xlabel("time  T")
    a1.set_ylabel("frequency  F")
    a1.set_xticks([])
    a1.set_yticks([])
    fig.text(0.605, 0.45, "unsqueeze(0)  →", fontsize=19, color=SECOND, va="center")
    a2 = fig.add_axes((0.75, 0.10, 0.22, 0.62))
    a2.imshow(spec, origin="lower", aspect="auto", cmap="cividis")
    a2.set_xticks([])
    a2.set_yticks([])
    a2.set_title("one channel: (1, F, T)", color=SECOND, loc="left")
    for s in a2.spines.values():
        s.set_color(SECOND)
        s.set_linewidth(3)
    save(fig, "audio_image")


# ---------------------------------------------------------------- nonlinearity, from the line up

SINE_X = torch.linspace(-3, 3, 200).unsqueeze(1)


def sine_data():
    true = torch.sin(3 * SINE_X)
    torch.manual_seed(1)
    return true, true + 0.15 * torch.randn_like(true)


def fit_sine(model, steps=2000, lr=1e-2):
    _, ys = sine_data()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(SINE_X) - ys) ** 2).mean()
        loss.backward()
        opt.step()
    return model, float(loss.detach())


def fig_line():
    x = np.linspace(-1, 4, 50)
    w, b = 0.8, 1.0
    fig, ax = plt.subplots(figsize=(8.4, 6))
    ax.axhline(0, color=GREY_FAINT, lw=1)
    ax.axvline(0, color=GREY_FAINT, lw=1)
    ax.plot(x, w * x + b, color=ACCENT, lw=3.5)
    ax.plot([1, 2, 2], [w + b, w + b, 2 * w + b], color=SECOND, lw=2.2)
    ax.text(
        2.08,
        1.5 * w + b,
        f"w = {w}\n(slope: up {w}\nfor every step right)",
        color=SECOND,
        va="center",
        fontsize=15,
    )
    ax.scatter([0], [b], s=90, color=THIRD, zorder=5)
    ax.text(
        -0.15,
        b + 0.25,
        f"b = {b:.0f}\n(height at x = 0)",
        color=THIRD,
        ha="right",
        fontsize=15,
    )
    ax.text(
        3.0, w * 3 + b + 0.55, "y = 0.8 x + 1", color=ACCENT, fontsize=20, ha="center"
    )
    ax.set_xlim(-2.2, 4.4)
    ax.set_ylim(-0.6, 4.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    save(fig, "line")


def fig_compose():
    x = np.linspace(-2, 3, 50)
    f1 = 2 * x + 1
    f2 = -0.5 * f1 + 3
    fig, ax = plt.subplots(figsize=(9.6, 6))
    ax.axhline(0, color=GREY_FAINT, lw=1)
    ax.axvline(0, color=GREY_FAINT, lw=1)
    ax.plot(x, f1, color=ACCENT_LIGHT, lw=3, label="f₁(x) = 2x + 1")
    ax.plot(x, -0.5 * x + 3, color=THIRD, lw=3, ls="--", label="f₂(u) = −0.5u + 3")
    ax.plot(x, f2, color=SECOND, lw=4, label="f₂(f₁(x)) = −x + 2.5")
    ax.legend(loc="upper left", fontsize=16)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    save(fig, "compose")


def fig_sine_two():
    """The sine test of notebook 02 §2.4: the linear stack draws one line, GELU draws the curve."""
    true, ys = sine_data()
    torch.manual_seed(0)
    lin, lin_loss = fit_sine(
        nn.Sequential(nn.Linear(1, 64), nn.Linear(64, 64), nn.Linear(64, 1))
    )
    torch.manual_seed(0)
    gel, gel_loss = fit_sine(
        nn.Sequential(
            nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1)
        )
    )
    l1, l2, l3 = lin
    with torch.no_grad():
        w = (l3.weight @ l2.weight @ l1.weight).item()
        b = (l3.weight @ (l2.weight @ l1.bias + l2.bias) + l3.bias).item()
    print(
        f"sine: linear stack MSE {lin_loss:.3f} = one line y = {w:.3f}x {b:+.3f}; with GELU {gel_loss:.3f}"
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    panels = [
        (
            axes[0],
            lin,
            f"Linear → Linear → Linear\n= one line: y = {w:.2f} x + {abs(b):.2f}",
            lin_loss,
            GREY_DIM,
        ),
        (axes[1], gel, "Linear → GELU → Linear → GELU → Linear", gel_loss, SECOND),
    ]
    with torch.no_grad():
        for ax, m, title, loss, c in panels:
            ax.scatter(SINE_X, ys, s=12, color=GREY_FAINT)
            ax.plot(SINE_X, true, color=GREY, lw=1.5, ls=":")
            ax.plot(SINE_X, m(SINE_X), color=c, lw=3.2)
            ax.set_title(f"{title}\nerror {loss:.3f}", color=c, fontsize=17)
            ax.set_xlabel("x")
    axes[0].set_ylim(-1.9, 1.7)
    save(fig, "sine_fit")


def fig_relu_units():
    """Notebook 02 §2.5: 5, 10 and 20 ReLU units, each a hinge; their sum follows the sine."""
    true, _ = sine_data()
    xs = SINE_X

    def relu_through(kx, ky):
        slopes = (ky[1:] - ky[:-1]) / (kx[1:] - kx[:-1])
        n = len(slopes)
        hinges, mix = nn.Linear(1, n), nn.Linear(n, 1)
        with torch.no_grad():
            hinges.weight.fill_(1.0)
            hinges.bias.copy_(-kx[:-1])
            mix.weight.copy_(torch.diff(slopes, prepend=torch.zeros(1)).unsqueeze(0))
            mix.bias.fill_(ky[0])
        return nn.Sequential(hinges, nn.ReLU(), mix)

    fig, axes = plt.subplots(
        2, 3, figsize=(16, 7.4), sharex=True, gridspec_kw={"hspace": 0.32}
    )
    for j, n in enumerate((5, 10, 20)):
        k = torch.linspace(-3, 3, n + 1)
        net = relu_through(k, torch.sin(3 * k))
        with torch.no_grad():
            parts = torch.relu(net[0](xs)) * net[2].weight
            total = net(xs)
        for u in range(n):
            axes[0, j].plot(
                xs,
                parts[:, u],
                color=CATEGORIES[u % len(CATEGORIES)],
                lw=1.4,
                alpha=0.85,
            )
        axes[0, j].set_title(f"{n} units: {n} hinges", color=GREY)
        axes[0, j].set_ylim(-14, 14)
        axes[1, j].plot(xs, true, color=GREY, lw=1.5, ls=":")
        axes[1, j].plot(xs, total, color=SECOND, lw=3)
        mse = float(((total - true) ** 2).mean())
        axes[1, j].set_title(f"their sum, error {mse:.4f}", color=SECOND)
        axes[1, j].set_ylim(-1.4, 1.4)
        axes[1, j].set_xlabel("x")
    axes[0, 0].set_ylabel("each unit")
    axes[1, 0].set_ylabel("the output")
    save(fig, "relu_units")


def fig_activations_grid():
    """Notebook 02 §2.6 per activation: the function and its slope, and the sine fit with it."""
    _, ys = sine_data()
    acts = {
        "ReLU": nn.ReLU(),
        "LeakyReLU": nn.LeakyReLU(0.1),
        "ELU": nn.ELU(),
        "GELU": nn.GELU(),
    }
    z = torch.linspace(-4, 4, 400, requires_grad=True)
    fig, axes = plt.subplots(
        2, 4, figsize=(15, 6.6), gridspec_kw={"hspace": 0.45, "wspace": 0.18}
    )
    for j, (name, act) in enumerate(acts.items()):
        c = SECOND if name == "GELU" else ACCENT
        y = act(z)
        (g,) = torch.autograd.grad(y.sum(), z)
        a = axes[0, j]
        a.axhline(0, color=GREY_FAINT, lw=1)
        a.axvline(0, color=GREY_FAINT, lw=1)
        a.plot(z.detach(), y.detach(), color=c, lw=3)
        a.plot(z.detach(), g, color=c, lw=1.8, ls="--")
        a.set_ylim(-1.5, 3)
        a.set_title(name, color=c, fontsize=19, fontweight="medium")
        a.set_xticks([])
        torch.manual_seed(0)
        model, loss = fit_sine(
            nn.Sequential(
                nn.Linear(1, 64), act, nn.Linear(64, 64), act, nn.Linear(64, 1)
            )
        )
        b = axes[1, j]
        with torch.no_grad():
            b.scatter(SINE_X, ys, s=6, color=GREY_FAINT)
            b.plot(SINE_X, model(SINE_X), color=c, lw=2.6)
        b.set_title(f"sine fit, error {loss:.3f}", color=c, fontsize=14)
        b.set_xticks([])
        b.set_ylim(-1.8, 1.8)
        if j:
            a.set_yticks([])
            b.set_yticks([])
        print(f"activation {name:<10} sine MSE {loss:.3f}")
    fig.text(
        0.125,
        0.94,
        "solid: the function   dashed: its slope",
        fontsize=15,
        color=GREY_DIM,
    )
    save(fig, "activations")


# ---------------------------------------------------------------- the loss, from a guess


LOSS_X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
LOSS_Y = np.array([2.1, 2.9, 4.2, 4.8, 6.1, 6.8])
GUESSES = [((0.3, 4.0), GREY_DIM, "guess 1"), ((0.9, 1.4), SECOND, "guess 2")]


def mse(w, b):
    return np.mean((w * LOSS_X + b - LOSS_Y) ** 2)


def fig_loss_guess():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    for ax, ((w, b), c, name) in zip(axes, GUESSES):
        pred = w * LOSS_X + b
        x = np.linspace(0.5, 6.5, 20)
        ax.plot(x, w * x + b, color=c, lw=3)
        for xi, yi, pi in zip(LOSS_X, LOSS_Y, pred):
            ax.plot([xi, xi], [yi, pi], color=c, lw=1.6, ls="--")
            ax.text(
                xi + 0.08,
                (yi + pi) / 2,
                f"{pi - yi:+.1f}",
                color=c,
                fontsize=13,
                va="center",
            )
        ax.scatter(LOSS_X, LOSS_Y, s=80, color=ACCENT, zorder=5)
        ax.set_title(
            f"{name}: ŷ = {w}x + {b}\nmean of the squared errors: {mse(w, b):.2f}",
            color=c,
            fontsize=18,
        )
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    print(
        f"loss guess: guess 1 MSE {mse(*GUESSES[0][0]):.2f}, guess 2 MSE {mse(*GUESSES[1][0]):.2f}"
    )
    save(fig, "loss_guess")


def fig_loss_landscape():
    gw, gb = np.meshgrid(np.linspace(-0.4, 2.0, 300), np.linspace(-1.5, 5.5, 300))
    Z = np.mean((gw[..., None] * LOSS_X + gb[..., None] - LOSS_Y) ** 2, axis=-1)
    A = np.stack([LOSS_X, np.ones_like(LOSS_X)], 1)
    w_best, b_best = np.linalg.lstsq(A, LOSS_Y, rcond=None)[0]
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    cs = ax.contourf(
        gw, gb, Z, levels=np.geomspace(0.02, 60, 18), cmap="Blues_r", alpha=0.9
    )
    ax.contour(gw, gb, Z, levels=np.geomspace(0.02, 60, 18), colors=BG, linewidths=0.6)
    for (w, b), c, name in GUESSES:
        ax.scatter([w], [b], s=140, color=c, ec=BG, lw=2, zorder=5)
        ax.text(
            w + 0.07,
            b + 0.25,
            f"{name}: {mse(w, b):.2f}",
            color=c,
            fontsize=16,
            bbox={"facecolor": BG, "edgecolor": "none", "pad": 3},
        )
    ax.scatter([w_best], [b_best], marker="*", s=420, color=TEXT, zorder=6)
    ax.text(
        w_best + 0.08,
        b_best - 0.6,
        f"lowest: {mse(w_best, b_best):.2f}",
        fontsize=16,
        bbox={"facecolor": BG, "edgecolor": "none", "pad": 3},
    )
    cb = fig.colorbar(cs, ax=ax, shrink=0.8, ticks=[0.1, 1, 10, 50])
    cb.set_label("loss (mean squared error)")
    cb.ax.set_yticklabels(["0.1", "1", "10", "50"])
    ax.set_xlabel("w (slope)")
    ax.set_ylabel("b (height)")
    print(
        f"loss landscape: lowest at w {w_best:.2f}, b {b_best:.2f}, MSE {mse(w_best, b_best):.3f}"
    )
    save(fig, "loss_landscape")


def fig_optimizer_valley():
    """Notebook 03 §3.3, left panel: one valley, three optimizers, 40 steps each."""

    def valley(w):
        return 0.1 * w[0] ** 2 + 2 * w[1] ** 2

    def descend(make, steps=40):
        w = torch.tensor([-4.0, 2.0], requires_grad=True)
        opt = make([w])
        path = [w.detach().clone()]
        for _ in range(steps):
            opt.zero_grad()
            valley(w).backward()
            opt.step()
            path.append(w.detach().clone())
        return torch.stack(path).numpy()

    runs = [
        ("SGD", descend(lambda p: torch.optim.SGD(p, lr=0.2)), GREY_DIM),
        (
            "SGD + momentum",
            descend(lambda p: torch.optim.SGD(p, lr=0.2, momentum=0.7)),
            THIRD,
        ),
        ("Adam", descend(lambda p: torch.optim.Adam(p, lr=0.3)), SECOND),
    ]
    gx, gy = np.meshgrid(np.linspace(-4.6, 1.2, 300), np.linspace(-2.4, 2.4, 300))
    Z = 0.1 * gx**2 + 2 * gy**2
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.contourf(
        gx, gy, Z, levels=np.geomspace(0.01, 14, 16), cmap="Blues_r", alpha=0.85
    )
    for name, p, c in runs:
        ax.plot(p[:, 0], p[:, 1], "-o", color=c, ms=4, lw=2, label=name)
    ax.scatter([-4], [2], s=120, color=TEXT, zorder=6)
    ax.text(-4, 2.15, "start", ha="center", fontsize=15)
    ax.scatter([0], [0], marker="X", s=160, color=TEXT, zorder=6)
    ax.text(
        0,
        0.6,
        "lowest loss",
        ha="center",
        va="bottom",
        fontsize=15,
        bbox={"boxstyle": "round,pad=0.25", "fc": BG, "ec": "none", "alpha": 0.9},
    )
    ax.set_axis_off()
    ax.legend(
        loc="lower right", fontsize=15, facecolor=BG, frameon=True, edgecolor=GREY_FAINT
    )
    save(fig, "optimizer")


# ---------------------------------------------------------------- learning curves and noise

CURVES = DATA / "dtd_curves.json"


class ResBlock(nn.Module):
    """x + Linear(act(Linear(norm(x)))): a block the signal can also skip."""

    def __init__(self, h: int, act: type[nn.Module]):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), act(), nn.Linear(h, h))

    def forward(self, x):
        return x + self.f(x)


def dtd_head(n: int, h: int, act: str, res: bool = False) -> nn.Module:
    acts = {"relu": nn.ReLU, "gelu": nn.GELU}
    if res:
        return nn.Sequential(
            nn.Linear(384, h),
            *[ResBlock(h, acts[act]) for _ in range(n)],
            nn.Linear(h, 47),
        )
    layers, w = [], 384
    for _ in range(n):
        layers += [nn.Linear(w, h), acts[act]()]
        w = h
    return nn.Sequential(*layers, nn.Linear(w, 47))


def dtd_curves(configs, epochs=100):
    """Train small heads on the DTD dinov2-small cache and keep every epoch's losses."""
    import time as _t

    Xtr, ytr = xy(cache("dtd-dinov2-small", "train"))
    Xva, yva = xy(cache("dtd-dinov2-small", "test"))
    out = {}
    for n, h, act, lr, *res in configs:
        torch.manual_seed(0)
        model = dtd_head(n, h, act, bool(res))
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        lf = nn.CrossEntropyLoss()
        rec = {"train": [], "valid": [], "acc": [], "sec": 0.0}
        t0 = _t.time()
        for _ in range(epochs):
            model.train()
            perm, tot = torch.randperm(len(Xtr)), 0.0
            for i in range(0, len(Xtr), 32):
                idx = perm[i : i + 32]
                opt.zero_grad()
                loss = lf(model(Xtr[idx]), ytr[idx])
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            model.eval()
            with torch.no_grad():
                o = model(Xva)
            rec["train"].append(tot / len(Xtr))
            rec["valid"].append(lf(o, yva).item())
            rec["acc"].append((o.argmax(1) == yva).float().mean().item())
        rec["sec"] = _t.time() - t0
        out["|".join([str(n), str(h), act, str(lr)] + (["res"] if res else []))] = rec
    return out


CURVE_CONFIGS = [
    (0, 0, "gelu", 1e-3),
    (0, 0, "gelu", 1e-4),
    (1, 512, "gelu", 1e-3),
    (1, 512, "gelu", 1e-4),
    (3, 512, "gelu", 3e-4),
    (3, 148, "relu", 3e-4),
    (3, 49, "gelu", 2e-4),
    (2, 512, "relu", 3e-5),
    (8, 128, "relu", 3e-5, "res"),
    (8, 256, "gelu", 1e-4, "res"),
]


def load_curves():
    if not CURVES.exists():
        CURVES.write_text(json.dumps(dtd_curves(CURVE_CONFIGS)), encoding="utf-8")
    return json.loads(CURVES.read_text(encoding="utf-8"))


def describe(key: str) -> str:
    n, h, act, lr, *res = key.split("|")
    if res:
        body = f"{n} residual blocks × {h}, {act}"
    else:
        body = "one Linear" if n == "0" else f"{n} hidden × {h}, {act}"
    return f"{body}, lr {float(lr):.0e}"


def fig_curve_kinds():
    """Three learning curves that end differently: still learning, finished, overfitting."""
    res = load_curves()
    picks = [
        ("8|128|relu|3e-05|res", "still learning: both curves still falling", False),
        ("1|512|gelu|0.0001", "valid stops improving: stop early", True),
        ("0|0|gelu|0.001", "overfitting: valid turns up", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"wspace": 0.22})
    for ax, (key, title, stop) in zip(axes, picks):
        r = res[key]
        e = np.arange(1, 31)
        ax.plot(e, r["train"][:30], color=ACCENT, lw=2.6, label="train")
        ax.plot(e, r["valid"][:30], color=SECOND, lw=2.6, label="valid")
        if stop:
            k = int(np.argmin(r["valid"][:30]))
            ax.scatter(
                [k + 1],
                [r["valid"][k]],
                s=170,
                color=SECOND,
                edgecolor=TEXT,
                linewidth=2,
                zorder=5,
            )
            ax.annotate(
                "early stopping",
                (k + 1, r["valid"][k]),
                xytext=(k + 5, r["valid"][k] * 1.6),
                fontsize=14,
                color=TEXT,
                arrowprops={"arrowstyle": "-", "color": TEXT, "lw": 1.2},
            )
        ax.set_title(title, color=GREY, fontsize=15)
        ax.set_xlabel("epoch")
        ax.set_ylim(0, None)
    axes[0].set_ylabel("loss")
    axes[0].legend(loc="upper right")
    save(fig, "curve_kinds")


def fig_search_curves():
    """Eight heads of the kind the handed-out search draws, 15 epochs each: the curves of E3."""
    res = load_curves()
    keys = sorted(res, key=lambda k: -res[k]["acc"][14])
    fig, axes = plt.subplots(
        2, 4, figsize=(18, 7.4), gridspec_kw={"hspace": 0.55, "wspace": 0.25}
    )
    for ax, key in zip(axes.flat, keys[:4] + keys[-4:]):
        r = res[key]
        e = np.arange(1, 16)
        ax.plot(e, r["train"][:15], color=ACCENT, lw=2.4, label="train")
        ax.plot(e, r["valid"][:15], color=SECOND, lw=2.4, label="valid")
        ax.set_title(
            f"accuracy {r['acc'][14]:.3f}\n{describe(key)}", color=GREY, fontsize=13
        )
        ax.set_xlabel("epoch")
    axes[0, 0].set_ylabel("loss")
    axes[1, 0].set_ylabel("loss")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=15, frameon=False)
    fig.subplots_adjust(right=0.9)
    save(fig, "search_curves")


def fig_longer():
    """The same heads trained for 100 epochs: the ranking at epoch 15 is not the ranking at 100."""
    res = load_curves()
    shown = {
        "1|512|gelu|0.001": SECOND,
        "2|512|relu|3e-05": ACCENT,
        "8|128|relu|3e-05|res": THIRD,
    }
    fig, ax = plt.subplots(figsize=(13, 6.4))
    e = np.arange(1, 101)
    for key, r in res.items():
        print(
            f"longer: {describe(key):<40} acc@15 {r['acc'][14]:.3f} acc@100 {r['acc'][-1]:.3f} best {max(r['acc']):.3f}  {r['sec']:.0f}s"
        )
        if key not in shown:
            ax.plot(e, r["acc"], color=GREY_FAINT, lw=1.4, zorder=1)
    ys = {
        "1|512|gelu|0.001": 0.797,
        "2|512|relu|3e-05": 0.824,
        "8|128|relu|3e-05|res": 0.770,
    }
    for key, c in shown.items():
        r = res[key]
        ax.plot(e, r["acc"], color=c, lw=3, zorder=3)
        ax.scatter([15], [r["acc"][14]], color=c, s=70, zorder=4)
        ax.text(
            102,
            ys[key],
            f"{describe(key)}\n{r['acc'][14]:.3f} at 15 → {r['acc'][-1]:.3f} at 100",
            color=c,
            va="center",
            fontsize=13,
        )
    ax.axvline(15, color=GREY, ls="--", lw=1.4)
    ax.text(16, 0.665, "the search stopped here", color=GREY, fontsize=14)
    ax.set_xlim(0, 140)
    ax.set_ylim(0.65, 0.835)
    ax.set_xlabel("epoch")
    ax.set_ylabel("valid accuracy")
    save(fig, "longer")


def fig_noise():
    """Why noise on the input helps: the same head on a few points, without and with jitter."""
    torch.manual_seed(3)
    n = 14
    X = torch.randn(2 * n, 2) * 0.9
    y = (X[:, 0] + 0.4 * torch.randn(2 * n) > 0).long()

    def train(sigma):
        torch.manual_seed(0)
        m = nn.Sequential(
            nn.Linear(2, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        for _ in range(1500):
            opt.zero_grad()
            nn.functional.cross_entropy(
                m(X + sigma * torch.randn_like(X)), y
            ).backward()
            opt.step()
        return m

    gx, gy = np.meshgrid(np.linspace(-2.6, 2.6, 300), np.linspace(-2.6, 2.6, 300))
    grid = torch.tensor(np.stack([gx.ravel(), gy.ravel()], 1), dtype=torch.float32)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    for ax, sigma in zip(axes, (0.0, 0.5)):
        m = train(sigma)
        with torch.no_grad():
            p = m(grid).softmax(1)[:, 1].reshape(gx.shape).numpy()
        ax.contourf(
            gx, gy, p, levels=[0, 0.5, 1], colors=["#d9e0e2", "#f3d6cc"], alpha=0.9
        )
        ax.contour(gx, gy, p, levels=[0.5], colors=GREY, linewidths=2)
        if sigma:
            torch.manual_seed(1)
            for _ in range(6):
                J = X + sigma * torch.randn_like(X)
                ax.scatter(
                    J[:, 0],
                    J[:, 1],
                    s=14,
                    c=[ACCENT if v == 0 else SECOND for v in y],
                    alpha=0.25,
                    ec="none",
                )
        ax.scatter(
            X[:, 0],
            X[:, 1],
            s=90,
            c=[ACCENT if v == 0 else SECOND for v in y],
            ec=BG,
            lw=1.5,
            zorder=5,
        )
        ax.set_title(
            "no noise: the border curls around single points"
            if not sigma
            else f"noise σ = {sigma}: a moved copy every step, a smoother border",
            color=GREY,
            fontsize=16,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    save(fig, "noise")


FIGURES = [
    fig_doublings,
    fig_flowers_pca_thumbs,
    fig_encoders_thumbs,
    fig_audio_image,
    fig_line,
    fig_compose,
    fig_sine_two,
    fig_relu_units,
    fig_activations_grid,
    fig_loss_guess,
    fig_loss_landscape,
    fig_optimizer_valley,
    fig_curve_kinds,
    fig_search_curves,
    fig_longer,
    fig_noise,
    fig_cover,
    fig_cat_channels,
    fig_samples,
    fig_batch_images,
    fig_sentences,
    fig_ladder,
    fig_unit_axis,
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
