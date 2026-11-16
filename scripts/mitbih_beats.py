"""Single heartbeats from the MIT-BIH Arrhythmia Database, split by patient.

The split is the inter-patient split of de Chazal, O'Dwyer and Reilly (2004), the
standard for evaluating heartbeat classifiers: DS1 trains, DS2 tests, and no patient has
beats on both sides. The four records with a pacemaker (102, 104, 107, 217) are in
neither, as in that paper.

A beat is cut the way Kachuee, Fazeli and Sarrafzadeh (2018) cut theirs, at 125 Hz:

1. the MLII lead, resampled from 360 Hz to 125 Hz, in windows of 10 seconds;
2. each window scaled to [0, 1] by its own minimum and maximum;
3. T, the median interval between the annotated R-peaks of the window;
4. from every annotated R-peak, a segment of 1.2 T, zero-padded at the end to 187 steps
   (cut at 187 when 1.2 T is longer).

The R-peaks and labels are the cardiologists' annotations, not a peak detector's. A beat
whose segment runs past the end of its window, or whose window holds fewer than two
beats, is left out. Each annotation symbol maps to an AAMI class (`AAMI`); symbols that
mark no beat are skipped.

Three classes are kept: N, S and V. The AAMI classes F (a fusion of V and N) and Q (paced
or unclassifiable) are left out: in this split nearly all F beats of train come from one
patient (record 208) and nearly all of test from one other (213), and Q is almost absent
once the paced records are out. Their beats still count for T, the median interval of
their window.

    uv run --with wfdb python scripts/mitbih_beats.py artefacts/sources/mitdb

prints the beats per class and split.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

FS_SOURCE = 360
FS = 125
STEPS = 187
WINDOW_SECONDS = 10
SEGMENT_PERIODS = 1.2

CLASSES = ["N", "S", "V"]

# annotation symbol -> AAMI class (ANSI/AAMI EC57), as in de Chazal et al. (2004)
AAMI = {
    **dict.fromkeys(["N", "L", "R", "e", "j"], "N"),
    **dict.fromkeys(["A", "a", "J", "S"], "S"),
    **dict.fromkeys(["V", "E"], "V"),
    "F": "F",
    **dict.fromkeys(["/", "f", "Q"], "Q"),
}

DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
       124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]  # fmt: skip
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
       212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]  # fmt: skip
SPLITS = {"train": DS1, "test": DS2}


def record_beats(
    source: Path, record: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The beats of one record: steps `(n, 187)` float32, labels `(n,)` as indices into
    `CLASSES`, and the R-peak of each beat as a sample index at 360 Hz in the record."""
    import wfdb  # ty: ignore[import-not-found]  (not a project dependency: `uv run --with wfdb`)

    path = str(source / str(record))
    signal = wfdb.rdrecord(path)
    lead = signal.sig_name.index("MLII")
    x = resample_poly(signal.p_signal[:, lead], FS, FS_SOURCE)
    ann = wfdb.rdann(path, "atr")

    keep = [i for i, s in enumerate(ann.symbol) if s in AAMI]
    peaks360 = np.asarray(ann.sample)[keep]
    # -1 for a beat of a class that is left out: it counts for T, it is not cut
    labels = np.array(
        [
            CLASSES.index(c) if (c := AAMI[ann.symbol[i]]) in CLASSES else -1
            for i in keep
        ]
    )
    peaks = np.round(peaks360 * FS / FS_SOURCE).astype(int)

    width = WINDOW_SECONDS * FS
    steps, kept_labels, kept_peaks = [], [], []
    for start in range(0, len(x) - width + 1, width):
        inside = (peaks >= start) & (peaks < start + width)
        if inside.sum() < 2:
            continue
        window = x[start : start + width]
        low, high = window.min(), window.max()
        if high <= low:
            continue
        scaled = (window - low) / (high - low)
        period = float(np.median(np.diff(peaks[inside])))
        length = min(round(SEGMENT_PERIODS * period), STEPS)
        for peak, label, peak360 in zip(
            peaks[inside], labels[inside], peaks360[inside], strict=True
        ):
            offset = peak - start
            if label < 0 or offset + length > width:
                continue
            beat = np.zeros(STEPS, dtype=np.float32)
            beat[:length] = scaled[offset : offset + length]
            steps.append(beat)
            kept_labels.append(label)
            kept_peaks.append(peak360)
    return (
        np.asarray(steps, dtype=np.float32).reshape(-1, STEPS),
        np.asarray(kept_labels, dtype=np.int64),
        np.asarray(kept_peaks, dtype=np.int64),
    )


def split_beats(source: Path, split: str) -> dict[str, np.ndarray]:
    """All beats of a split, record by record in the order of `SPLITS`, each record's
    beats in time order: `steps`, `label`, `record` and `r_peak` (360 Hz)."""
    parts = {"steps": [], "label": [], "record": [], "r_peak": []}
    for record in SPLITS[split]:
        steps, labels, peaks = record_beats(source, record)
        parts["steps"].append(steps)
        parts["label"].append(labels)
        parts["record"].append(np.full(len(labels), record, dtype=np.int64))
        parts["r_peak"].append(peaks)
    return {k: np.concatenate(v) for k, v in parts.items()}


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artefacts/sources/mitdb")
    for split, records in SPLITS.items():
        beats = split_beats(source, split)
        counts = Counter(beats["label"].tolist())
        total = len(beats["label"])
        shares = ", ".join(
            f"{c} {counts.get(i, 0)} ({counts.get(i, 0) / total:.1%})"
            for i, c in enumerate(CLASSES)
        )
        print(f"{split}: {total} beats from {len(records)} records; {shares}")


if __name__ == "__main__":
    main()
