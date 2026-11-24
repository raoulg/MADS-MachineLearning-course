"""Measure and draw the lesson 2 slides on depth and the skip, and on the S beat.

Draw the checked measurements::

    uv run scripts/lesson2_evidence.py

Measure the fixed notebook designs on the full train/validation data and draw::

    uv run --with wfdb scripts/lesson2_evidence.py --measure

Raw MIT-BIH files live in artefacts/sources/mitdb. The notebook supplies the model
classes; its source is read without changing or executing the notebook in place.
"""

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "presentations/img/lesson2"
DATA = DEST / "lesson2-evidence.json"
DEST.mkdir(parents=True, exist_ok=True)
P, V, I, M, R, BG = "#1d5a73", "#c43a14", "#111111", "#5a544d", "#cfc7bb", "#f4f1ec"
font = base64.b64encode(
    (ROOT / "presentations/fonts/Outfit-400.woff2").read_bytes()
).decode()


def text(x, y, s, size=28, c=I, anchor="start"):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{c}" text-anchor="{anchor}">{html.escape(str(s))}</text>'


def line(x, y, xx, yy, c=R, w=2, dash=""):
    return (
        f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{xx:.1f}" y2="{yy:.1f}" stroke="{c}" stroke-width="{w}"'
        + (f' stroke-dasharray="{dash}"' if dash else "")
        + "/>"
    )


def path(x, y, c=P, w=3, dash=""):
    points = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(x, y, strict=True))
    return (
        f'<polyline points="{points}" fill="none" stroke="{c}" stroke-width="{w}" stroke-linejoin="round"'
        + (f' stroke-dasharray="{dash}"' if dash else "")
        + "/>"
    )


def rect(x, y, w, h, fill="none", stroke=R):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}"/>'


def save(name, body, label):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1720 600" role="img" aria-label="{html.escape(label)}"><style>@font-face{{font-family:Outfit;src:url(data:font/woff2;base64,{font})}}text{{font-family:Outfit,Arial,sans-serif}}</style><rect width="1720" height="600" fill="{BG}"/>{body}</svg>'
    (DEST / f"{name}.svg").write_text(svg)


SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def power(e):
    return {0: "1", -1: "0.1", -2: "0.01", -3: "0.001"}.get(
        e, "10" + str(e).translate(SUP)
    )


def diagnostics(data, key, name, ylabel):
    lo, hi = (-10, 0) if key == "spread" else (-9, 0)
    x = lambda a: 260 + np.asarray(a) / 16 * 1150
    y = lambda a: 470 - (np.log10(np.maximum(a, 1e-12)) - lo) / (hi - lo) * 360
    s = text(60, 36, ylabel, 30, M)
    for e in range(hi if (hi - lo) % 2 == 0 else hi - 1, lo - 1, -2):
        yy = float(y(10.0**e))
        s += line(260, yy, 1410, yy) + text(240, yy + 9, power(e), 26, M, "end")
    s += line(260, 470, 1410, 470, M)
    first = 1 if key == "gradient" else 0
    for j in range(first, 17):
        if j in (first, 4, 8, 12, 16):
            label = "before 1" if j == 0 else str(j)
            s += text(float(x(j)), 505, label, 26, M, "middle")
    s += text(835, 545, "after block" if key == "spread" else "block", 28, M, "middle")
    ends = {}
    for arm, col, dash in [("plain", M, "8 7"), ("skip", P, "")]:
        values = np.array(data["diagnostics"][arm][key])
        xx = np.arange(len(values)) + first
        s += path(x(xx), y(values), col, 5, dash)
        for a, b in zip(x(xx), y(values), strict=True):
            s += f'<circle cx="{a:.1f}" cy="{b:.1f}" r="5" fill="{col}"/>'
        ends[arm] = (values, xx)
    pv = ends["plain"][0]
    sv = ends["skip"][0]
    if key == "spread":
        s += text(1430, float(y(sv[-1])) + 9, f"with skip · {sv[-1]:.2f}", 28, P)
        s += text(1430, float(y(pv[-1])) - 6, "plain · 7 × 10⁻¹⁰", 28, M)
        s += text(1430, float(y(pv[-1])) + 28, "all beats alike", 26, M)
    else:
        s += text(1430, float(y(sv[-1])) + 9, "with skip", 28, P)
        s += text(1430, float(y(pv[-1])) + 30, "plain", 28, M)
        s += line(float(x(1)) + 8, float(y(pv[0])) + 4, float(x(4)) - 8, 452, M, 1)
        s += text(float(x(4)), 458, "block 1: 6 × 10⁻⁹, it barely moves", 26, M)
        s += (
            line(float(x(15)), 88, float(x(2)), 88, V, 3)
            + line(float(x(2)), 88, float(x(2)) + 14, 80, V, 3)
            + line(float(x(2)), 88, float(x(2)) + 14, 96, V, 3)
        )
        s += text(
            float(x(8.5)),
            76,
            "backward pass: from the head to block 1",
            26,
            V,
            "middle",
        )
        s += text(float(x(1)), 590, "input side", 22, M, "middle") + text(
            float(x(16)), 590, "head side", 22, M, "middle"
        )
    save(name, s, ylabel + " before training, plain versus skip, sixteen blocks")


ZOOM = (4.55, 6.95)


def beat_times(a):
    peaks = np.array(a["peaks"])
    times = peaks / 360 - a["start_seconds"]
    j = int(np.flatnonzero(peaks == a["target_sample"])[0])
    return times, j


def bracket(x0, x1, yy, label, c, size=30):
    return (
        line(x0, yy, x1, yy, c, 3)
        + line(x0, yy - 10, x0, yy + 10, c, 3)
        + line(x1, yy - 10, x1, yy + 10, c, 3)
        + text((x0 + x1) / 2, yy - 16, label, size, c, "middle")
    )


def rhythm(d):
    a = d["ecg"]
    x = lambda t: 80 + np.asarray(t) * 156
    signal = np.array(a["signal"])
    y = lambda v: 360 - np.asarray(v) * 160
    times, j = beat_times(a)
    s = text(80, 50, "one patient, ten seconds", 30, M)
    s += rect(x(ZOOM[0]), 95, x(ZOOM[1]) - x(ZOOM[0]), 420, "#e8e2d9", "none")
    s += text(x(ZOOM[1]) - 12, 125, "next slide", 26, M, "end")
    for t in range(11):
        s += line(x(t), 150, x(t), 500, R, 1) + text(x(t), 546, t, 26, M, "middle")
    s += path(x(np.arange(len(signal)) / 360), y(signal), P, 2)
    for k, (t, symbol) in enumerate(zip(times, a["symbols"], strict=True)):
        target = k == j
        s += text(x(t), 175, "S" if target else "N", 30, V if target else M, "middle")
    s += text(860, 594, "seconds into this excerpt", 28, M, "middle")
    save(
        "s-rhythm",
        s,
        "Ten seconds of ECG from record 209: fifteen beats labelled N and one labelled S; four beats around S are marked for the next slide",
    )


def zoom(d):
    a = d["ecg"]
    lo, hi = ZOOM
    x = lambda t: 110 + (np.asarray(t) - lo) / (hi - lo) * 1500
    signal = np.array(a["signal"])
    y = lambda v: 385 - np.asarray(v) * 150
    times, j = beat_times(a)
    t = np.arange(len(signal)) / 360
    keep = (t >= lo) & (t <= hi)
    s = ""
    for sec in np.arange(5, 7, 0.5):
        s += line(x(sec), 250, x(sec), 530, R, 1) + text(
            x(sec), 568, f"{sec:g}", 26, M, "middle"
        )
    s += path(x(t[keep]), y(signal[keep]), P, 3)
    beats = [j - 2, j - 1, j, j + 1]
    for k in beats:
        c = V if k == j else M
        s += line(x(times[k]), 180, x(times[k]), 540, c, 2, "6 5")
        s += text(x(times[k]), 168, "S" if k == j else "N", 34, c, "middle")
    for (l, r), c, label in [
        ((j - 2, j - 1), M, "0.64 s"),
        ((j - 1, j), V, "0.50 s · early"),
        ((j, j + 1), M, "0.71 s · longer pause"),
    ]:
        s += bracket(x(times[l]), x(times[r]), 105, label, c)
    s += text(40, 40, "time between peaks", 30, M)
    s += text(860, 598, "seconds", 26, M, "middle")
    save(
        "s-zoom",
        s,
        "Four beats N, N, S, N: the gaps between peaks are 0.64 s, then 0.50 s before S, then 0.71 s after S",
    )


def crop(data):
    a = data["ecg"]
    times, j = beat_times(a)
    s_t = times[j]
    lo, hi = ZOOM[0], s_t + 187 / 125 + 0.05
    x = lambda t: 110 + (np.asarray(t) - lo) / (hi - lo) * 1500
    signal = np.array(a["signal"])
    t = np.arange(len(signal)) / 360
    keep = (t >= lo) & (t <= hi)
    y = lambda v: 220 - np.asarray(v) * 90
    row_end = s_t + 96 / 125
    s = text(40, 34, "the recording", 30, M)
    s += rect(x(s_t), 80, x(row_end) - x(s_t), 230, "none", V)
    s += path(x(t[keep]), y(signal[keep]), P, 2)
    for k in range(len(times)):
        if lo <= times[k] <= hi:
            c = V if k == j else M
            s += text(x(times[k]), 72, "S" if k == j else "N", 30, c, "middle")
    s += bracket(x(times[j - 1]), x(s_t), 365, "0.50 s · not in the row", V, 28)
    yy = lambda v: 520 - np.asarray(v) * 110
    s += text(40, 440, "the row the model gets", 30, M)
    row = np.array(a["s_crop"])
    rx = s_t + np.arange(187) / 125
    s += line(x(s_t), 525, x(rx[-1]), 525, R, 1)
    s += path(x(rx), yy(row), V, 3)
    s += line(x(s_t), 310, x(s_t), 560, V, 2, "6 5")
    s += line(x(row_end), 310, x(row_end), 430, V, 1, "4 6")
    s += text(x(s_t) - 12, 585, "starts at the S peak", 26, V, "end")
    s += bracket(x(s_t), x(times[j + 1]), 580, "0.71 s · in the row", M, 26)
    s += text(
        (x(row_end) + x(rx[-1])) / 2,
        500,
        "zeros: padding to 187 values",
        26,
        M,
        "middle",
    )
    save(
        "s-crop",
        s,
        "The model row for the S beat starts at the S peak: 96 values of signal, then zeros up to 187; the 0.50 s gap before S is not in the row",
    )


def question(data):
    a = data["ecg"]
    times, j = beat_times(a)
    s_t, prev = times[j], times[j - 1]
    x = lambda t: 600 + np.asarray(t) / 10 * 1060
    signal = np.array(a["signal"])
    t = np.arange(len(signal)) / 360
    y = lambda v: 150 - np.asarray(v) * 55
    end = s_t + 96 / 125
    s = text(40, 40, "one 10-second window", 26, M) + text(
        40, 74, "of the published recordings", 26, M
    )
    s += path(x(t), y(signal), P, 1.5)
    s += text(x(s_t), 52, "S", 26, V, "middle")
    for k in range(len(times)):
        s += line(x(times[k]), 215, x(times[k]), 590, V if k == j else R, 1, "4 6")

    def bar(yy, start, stop, c, dash=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<rect x="{x(start):.1f}" y="{yy - 17}" width="{x(stop) - x(start):.1f}" height="34" fill="{c}" fill-opacity="{0 if dash else 0.18}" stroke="{c}" stroke-width="2"{d}/>'

    rows = [
        ("today: one beat, from its peak", V, [(s_t, end)]),
        ("two beats", P, [(prev - 0.1, end)]),
        ("ten seconds", P, [(0, 10)]),
    ]
    for i, (label, c, spans) in enumerate(rows):
        yy = 260 + i * 70
        s += text(40, yy + 10, label, 29, c)
        for start, stop in spans:
            s += bar(yy, start, stop, c)
    yy = 260 + 3 * 70
    s += text(40, yy + 10, "one beat + what is normal", 29, P)
    s += text(40, yy + 44, "for this patient", 29, P)
    s += bar(yy, s_t, end, V)
    feats = ["usual gap", "gap spread", "usual shape"]
    fx = x(end) + 24
    s += text(fx - 4, yy + 10, "+", 30, P)
    for f, label in enumerate(feats):
        s += rect(fx + 22 + f * 118, yy - 17, 110, 34, "none", P) + text(
            fx + 77 + f * 118, yy + 8, label, 20, P, "middle"
        )
    s += text(fx - 4, yy + 50, "features from 1 minute of this patient", 22, M)
    yy = 260 + 4 * 70 + 20
    s += text(40, yy + 10, "your idea", 29, M)
    s += bar(yy, 0, 10, M, "8 8")
    s += text(x(5), yy + 9, "?", 28, M, "middle")
    save(
        "s-question",
        s,
        "Ideas for what to give the model, on one ten-second window: today one beat from its peak, two beats, ten seconds, one beat plus features of what is normal for this patient, or your own idea",
    )


def main():
    d = json.loads(DATA.read_text())
    diagnostics(d, "spread", "depth-spread", "spread across the 256 beats, log scale")
    diagnostics(
        d,
        "gradient",
        "depth-gradient",
        "gradient length on a block's weights = √(sum of all its squared gradients), log scale",
    )
    rhythm(d)
    zoom(d)
    crop(d)
    question(d)
    print("6 figures drawn from", DATA.relative_to(ROOT))


def measure_evidence():
    import ast
    import json
    import os
    import sys
    from pathlib import Path

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mads-mpl")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import numpy as np
    import torch
    import wfdb  # ty: ignore[import-not-found]  (optional: uv run --with wfdb)
    from torch import nn
    from vectormesh.components import Conv1D, Gate, Pool1D, Skip

    ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [
        str(ROOT / "notebooks/2_kernels_state_and_gates"),
        str(ROOT / "scripts"),
    ]
    import heartbeats as hb
    from mitbih_beats import record_beats

    torch.set_num_threads(4)
    notebook = json.loads(
        (
            ROOT / "notebooks/2_kernels_state_and_gates/02_state_and_gates.ipynb"
        ).read_text()
    )
    scope: dict[str, Any] = {
        "hb": hb,
        "nn": nn,
        "torch": torch,
        "np": np,
        "Conv1D": Conv1D,
        "Gate": Gate,
        "Pool1D": Pool1D,
        "Skip": Skip,
    }
    for i in [8, 11, 12, 16, 35]:
        tree = ast.parse("".join(notebook["cells"][i]["source"]))
        tree.body = [
            node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        ]
        # Definitions come from the fixed course notebook in this checkout.
        exec(compile(tree, f"notebook-cell-{i}", "exec"), scope)  # noqa: S102
    train, valid = hb.load()
    print("loaded", len(train), len(valid), flush=True)
    Deep, plain_skip = scope["Deep"], scope["plain_skip"]
    scope["train"] = train
    torch.manual_seed(0)
    inside = {}
    for name, make in [
        ("plain", lambda: Deep(16)),
        ("skip", lambda: Deep(16, plain_skip)),
    ]:
        spread, size, gradient = scope["look_inside"](
            make(), train.beats[:256], train.labels[:256]
        )
        inside[name] = {"spread": spread, "size": size, "gradient": gradient}

    def serialise(result):
        return {
            "mean": float(result.mean),
            "std": float(result.std),
            "runs": [
                {
                    "scores": r.scores,
                    "train_loss": r.train_loss,
                    "val_loss": r.val_loss,
                    "val_recall": r.val_recall,
                }
                for r in result.runs
            ],
        }

    torch.manual_seed(0)
    deep = {}
    for name, make in [
        ("2 plain", lambda: Deep(2)),
        ("16 plain", lambda: Deep(16)),
        ("16 skip", lambda: Deep(16, plain_skip)),
    ]:
        found = hb.spread(make, train, valid, 10, 3)
        deep[name] = serialise(found)
        print(name, deep[name]["mean"], deep[name]["std"], flush=True)

    ThreeBlocks = scope["ThreeBlocks"]
    models = []
    gate = {}
    for name, gated in [("no gate", False), ("gate", True)]:

        def make(gated=gated):
            m = ThreeBlocks(gated)
            if gated:
                models.append(m)
            return m

        found = hb.spread(make, train, valid, 12, 3)
        gate[name] = serialise(found)
        print(name, gate[name]["mean"], gate[name]["std"], flush=True)
    model = models[0].cpu().eval()
    with torch.no_grad():
        features = model.features(valid.beats)
        every = torch.sigmoid(model.gate.project(features))
        inside_beat = (
            valid.beats[:, :184, 0].reshape(len(valid), 23, 8).abs().sum(-1) > 0
        )
        order = every[inside_beat].mean(dim=0).argsort()
        maps = {}
        for k, name in [(0, "N"), (2, "V")]:
            i = int(torch.where(valid.labels == k)[0][0])
            values = every[i][:, order].T.numpy().copy()
            magnitude = features[i][:, order].T.abs().numpy()
            values[:, ~inside_beat[i].numpy()] = np.nan
            values[magnitude < np.quantile(magnitude, 0.25)] = np.nan
            maps[name] = {
                "beat": valid.beats[i, :, 0].tolist(),
                "values": [
                    [None if np.isnan(v) else float(v) for v in row] for row in values
                ],
            }

    source = ROOT / "artefacts/sources/mitdb"
    record = wfdb.rdrecord(str(source / "209"))
    ann = wfdb.rdann(str(source / "209"), "atr")
    peaks = np.asarray(ann.sample)
    symbols = np.asarray(ann.symbol)
    beat = np.isin(
        symbols,
        ["N", "L", "R", "e", "j", "A", "a", "J", "S", "V", "E", "F", "/", "f", "Q"],
    )
    peaks, symbols = peaks[beat], symbols[beat]
    target = 545731
    j = int(np.flatnonzero(peaks == target)[0])
    assert symbols[j] == "A" and all(symbols[j - 4 : j] == "N")
    assert 209 not in hb.VAL_RECORDS
    steps, labels, kept = record_beats(source, 209)
    s = int(np.flatnonzero(kept == target)[0])
    n = int(np.flatnonzero(kept == peaks[j - 1])[0])
    assert labels[s] == 1 and labels[n] == 0
    fs = 360
    start = 1510
    local = (peaks >= start * fs) & (peaks < (start + 10) * fs)
    ecg = {
        "record": 209,
        "lead": "MLII",
        "fs": fs,
        "start_seconds": start,
        "window_seconds": 10,
        "target_sample": target,
        "normal_sample": int(peaks[j - 1]),
        "signal": record.p_signal[
            start * fs : (start + 10) * fs, record.sig_name.index("MLII")
        ].tolist(),
        "peaks": peaks[local].tolist(),
        "symbols": symbols[local].tolist(),
        "preceding_rr_seconds": float((peaks[j] - peaks[j - 1]) / fs),
        "reference_rr_seconds": float(np.median(np.diff(peaks[j - 4 : j])) / fs),
        "normal_crop": steps[n].tolist(),
        "s_crop": steps[s].tolist(),
        "metadata": record.comments,
    }
    out = DATA
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "protocol": {
                    "train_counts": train.counts(),
                    "valid_counts": valid.counts(),
                    "repeats": 3,
                    "deep_epochs": 10,
                    "gate_epochs": 12,
                    "device": str(hb.device()),
                    "score": "mean validation macro recall over last 3 epochs",
                    "dataset_revision": hb.REVISION,
                },
                "diagnostics": inside,
                "deep": deep,
                "gate": gate,
                "maps": maps,
                "ecg": ecg,
            },
            indent=2,
        )
    )
    print(out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measure",
        action="store_true",
        help="Run the fixed notebook experiments before drawing",
    )
    if parser.parse_args().measure:
        measure_evidence()
    main()
