"""Reproduce and draw the fixed experiments used by the lesson 2 slide preview.

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
DEST = ROOT / "presentations/img/lesson2/gates-s"
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


def legend(x, y, items):
    s = ""
    for label, c, dash in items:
        s += line(x, y - 9, x + 55, y - 9, c, 4, dash) + text(x + 70, y, label, 30, c)
        x += 330
    return s


def diagnostics(data, key, name, ylabel):
    x = lambda a: 180 + np.asarray(a) / 16 * 1330
    ymin, ymax = (-10, 0) if key == "spread" else (-10, 2)
    y = lambda a: 455 - (np.log10(np.maximum(a, 1e-12)) - ymin) / (ymax - ymin) * 345
    s = legend(500, 42, [("plain", M, "7 7"), ("with skip", P, "")])
    s += text(180, 85, ylabel, 29, M) + text(1510, 85, "log scale", 25, M, "end")
    for e in range(ymin, ymax + 1, 2):
        yy = float(y(10.0**e))
        s += line(180, yy, 1510, yy) + text(150, yy + 9, f"1e{e}", 25, M, "end")
    for j in [0, 4, 8, 12, 16]:
        s += text(float(x(j)), 500, j, 28, M, "middle")
    s += line(180, 455, 1510, 455, M) + text(
        845,
        550,
        "block (1 = nearest input)"
        if key == "gradient"
        else "block (0 = before blocks)",
        29,
        M,
        "middle",
    )
    for arm, col, dash in [("plain", M, "7 7"), ("skip", P, "")]:
        values = data["diagnostics"][arm][key]
        xx = np.arange(len(values)) + (1 if key == "gradient" else 0)
        s += path(x(xx), y(np.array(values)), col, 5, dash)
        for a, b in zip(x(xx), y(np.array(values)), strict=True):
            s += f'<circle cx="{a}" cy="{b}" r="5" fill="{col}"/>'
    if key == "gradient":
        s += text(180, 590, "input side", 24, M) + text(
            1510, 590, "head side", 24, M, "end"
        )
    save(name, s, ylabel + " before training: plain versus skip across sixteen blocks")


def scores(data, key, name, items):
    x = lambda v: 610 + (v - 0.25) / 0.4 * 970
    s = text(610, 60, "validation macro recall", 31, M) + text(
        610, 560, "Mean ± std across 3 runs", 28, M
    )
    for j, (label, k, col) in enumerate(items):
        yy = 155 + j * 125
        r = data[key][k]
        mean, std = r["mean"], r["std"]
        s += text(80, yy + 12, label, 38, col)
        s += line(x(mean - std), yy, x(mean + std), yy, col, 5)
        for v in [mean - std, mean + std]:
            s += line(x(v), yy - 13, x(v), yy + 13, col, 3)
        s += f'<circle cx="{x(mean)}" cy="{yy}" r="11" fill="{col}"/>'
        s += text(1660, yy + 12, f"{mean:.3f}", 32, col, "end")
    s += line(610, 465, 1580, 465, M)
    for v in [0.3, 0.4, 0.5, 0.6]:
        s += line(x(v), 465, x(v), 477, M) + text(
            x(v), 515, f"{v:.1f}", 28, M, "middle"
        )
    save(
        name,
        s,
        "Repeated validation macro recall, shared scale, whiskers are run standard deviations",
    )


def heatmaps(data):
    s = ""
    for name, left in [("N", 105), ("V", 965)]:
        d = data["maps"][name]
        s += text(left, 34, f"one {name} beat", 34, P)
        a = np.asarray(d["beat"])
        s += path(left + np.arange(185) / 125 / 1.472 * 630, 155 - a[:185] * 90, P, 2)
        s += text(left, 202, "gate multipliers", 27, M)
        for r, row in enumerate(d["values"]):
            for c, v in enumerate(row):
                if v is None:
                    fill = "#ffffff"
                else:
                    low = np.array([20, 49, 67])
                    high = np.array([212, 230, 232])
                    rgb = (low + (high - low) * v).astype(int)
                    fill = "#" + "".join(f"{v:02x}" for v in rgb)
                s += rect(
                    left + c * 630 / 23, 225 + r * 8, 630 / 23 + 0.4, 8.2, fill, fill
                )
        for t in [0, 0.5, 1, 1.47]:
            s += text(left + t / 1.472 * 630, 527, f"{t:g}", 24, M, "middle")
        s += text(left + 315, 575, "seconds after R-peak", 28, M, "middle")
    s += text(20, 360, "channel", 22, M)
    s += rect(745, 155, 20, 20, "#ffffff", R) + text(775, 174, "masked", 24, M)
    for j in range(100):
        v = j / 99
        low = np.array([20, 49, 67])
        high = np.array([212, 230, 232])
        rgb = (low + (high - low) * v).astype(int)
        c = "#" + "".join(f"{v:02x}" for v in rgb)
        s += rect(670 + j * 3.5, 65, 3.6, 17, c, c)
    s += (
        text(640, 82, "0", 24, M, "end")
        + text(1040, 82, "1", 24, M)
        + text(845, 119, "closed → open", 24, M, "middle")
    )
    save(
        "gate-observed",
        s,
        "Two actual beat traces with their trained gate maps; dark is closed, pale is open, white is masked",
    )


def rhythm(d):
    a = d["ecg"]
    x = lambda t: 80 + np.asarray(t) * 156
    signal = np.array(a["signal"])
    y = lambda v: 385 - np.asarray(v) * 160
    s = text(80, 40, "Record 209 · MLII · one patient", 32, M)
    s += text(1640, 40, "atrial premature beat: A → S", 30, V, "end")
    for t in range(11):
        s += line(x(t), 170, x(t), 500, R, 1) + text(x(t), 546, t, 26, M, "middle")
    s += path(x(np.arange(len(signal)) / 360), y(signal), P, 2)
    for p, symbol in zip(a["peaks"], a["symbols"], strict=True):
        t = p / 360 - a["start_seconds"]
        target = p == a["target_sample"]
        s += text(
            x(t), 180, "S" if target else symbol, 30, V if target else M, "middle"
        )
        if target:
            s += line(x(t), 190, x(t), 500, V, 2, "6 5")
    peaks = np.array(a["peaks"])
    j = int(np.flatnonzero(peaks == a["target_sample"])[0])
    times = peaks / 360 - a["start_seconds"]
    for l, r, yy, label, c in [
        (times[j - 2], times[j - 1], 110, "0.64 s", M),
        (times[j - 1], times[j], 75, "0.50 s", V),
    ]:
        s += (
            line(x(l), yy, x(r), yy, c, 3)
            + line(x(l), yy - 8, x(l), yy + 8, c, 3)
            + line(x(r), yy - 8, x(r), yy + 8, c, 3)
            + text((x(l) + x(r)) / 2, yy - 15, label, 28, c, "middle")
        )
    s += text(860, 594, "seconds within the ten-second window", 28, M, "middle")
    save(
        "s-rhythm",
        s,
        "Real ten-second ECG window from record 209 at 1510 seconds; an A-annotated S beat has a preceding interval of 0.50 seconds versus 0.64 nearby",
    )


def crop(data):
    a = data["ecg"]
    target_t = a["target_sample"] / 360 - a["start_seconds"]
    signal = np.array(a["signal"])
    t = np.arange(len(signal)) / 360 - target_t
    keep = (t >= -0.8) & (t <= 0.85)
    x = lambda v: 170 + (np.asarray(v) + 0.8) / 1.65 * 1380
    y = lambda v: 225 - np.asarray(v) * 90
    s = text(70, 38, "The same S beat, with context", 32, M)
    s += rect(x(-0.8), 65, x(0) - x(-0.8), 210, "#e2dcd3", "none")
    s += path(x(t[keep]), y(signal[keep]), P, 2)
    s += line(x(0), 65, x(0), 288, V, 3, "6 5") + text(
        x(0) + 15, 105, "row starts at R", 28, V
    )
    s += text(x(-0.72), 105, "preceding RR + P region", 28, M)
    for v in [-0.5, 0, 0.5]:
        s += text(x(v), 315, f"{v:g} s", 24, M, "middle")
    s += text(70, 370, "Actual dataset rows", 32, M) + legend(
        840, 370, [("normal N", P, ""), ("target S", V, "7 6")]
    )
    xx = 170 + np.arange(187) / 125 / 1.496 * 1380
    for key, c, dash in [("normal_crop", P, ""), ("s_crop", V, "7 6")]:
        s += path(xx, 525 - np.array(a[key]) * 100, c, 3, dash)
    s += line(170, 535, 1550, 535, M)
    s += text(1330, 505, "zero padding", 27, M, "middle")
    for v in [0, 0.5, 1, 1.496]:
        s += text(170 + v / 1.496 * 1380, 571, f"{v:.1f}", 25, M, "middle")
    s += text(
        860,
        590,
        "seconds after R-peak",
        25,
        M,
        "middle",
    )
    save(
        "s-crop",
        s,
        "The same S beat before and after the course extraction, compared with the preceding N row; omitted left context includes preceding RR and P-wave region",
    )


def question():
    s = rect(610, 30, 500, 90, "none", P) + text(
        860, 88, "One hard S beat", 38, P, "middle"
    )
    for xx, label in [(280, "Information"), (860, "Learning"), (1440, "Decision")]:
        s += (
            line(860, 120, xx, 230, M, 3)
            + rect(xx - 220, 230, 440, 145, "none", P)
            + text(xx, 315, label, 38, P, "middle")
            + line(xx, 375, 860, 480, M, 3)
        )
    s += rect(370, 480, 980, 115, "none", V)
    for xx, label, explanation in [
        (535, "S recall", "of S beats, found"),
        (860, "S precision", "of S flags, correct"),
        (1185, "False S alarms", "of N, flagged S"),
    ]:
        s += text(xx, 525, label, 31, I, "middle")
        s += text(xx, 570, explanation, 25, M, "middle")
    save(
        "s-question",
        s,
        "Three open intervention points on one problem, feeding a shared evaluation of S detection and false alarms",
    )


def main():
    d = json.loads(
        (ROOT / "presentations/img/lesson2/gates-s/measurements.json").read_text()
    )
    diagnostics(d, "spread", "depth-spread", "spread across beats (std)")
    diagnostics(d, "gradient", "depth-gradient", "weight gradient size (norm)")
    scores(
        d,
        "deep",
        "depth-score",
        [
            ("2 plain blocks", "2 plain", P),
            ("16 plain blocks", "16 plain", M),
            ("16 skip blocks", "16 skip", P),
        ],
    )
    heatmaps(d)
    scores(
        d,
        "gate",
        "gate-score",
        [("3 blocks, no gate", "no gate", M), ("3 blocks + gate", "gate", P)],
    )
    rhythm(d)
    crop(d)
    question()
    (DEST / "measurements.json").write_text(json.dumps(d, indent=2))
    print("8 figures and measurements exported")


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
    out = ROOT / "presentations/img/lesson2/gates-s/measurements.json"
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
