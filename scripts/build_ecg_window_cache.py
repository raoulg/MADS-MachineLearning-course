"""Build the MIT-BIH ECG window VectorCaches, one per split, ready to publish.

    uv run --with wfdb python scripts/build_ecg_window_cache.py

Reads the PhysioNet records in artefacts/sources/mitdb (`make build-ecg-windows`
downloads them when they are missing) and cuts every record of a split into consecutive
windows of 10 seconds: the MLII lead, resampled from 360 Hz to 125 Hz, in millivolt,
unscaled. A row is one window, a float32 `(1250, 1)` tensor (sequence, channel), with the
annotated beats inside it. The split is the patient split of `mitbih_beats.py`: DS1
trains, DS2 tests, the paced records are in neither.

The windows are the ones `mitbih_beats.py` cuts its beats from: the same resampled lead,
the same start every 1250 steps from the first sample, the same beats inside. The tail
of a record after its last whole window (about 5.6 s) is left out. A window holds every
annotated beat, of all five AAMI classes, with no beat left out at the edges.

Columns, per window:

    ecg           float32 (1250, 1)  the MLII lead in mV at 125 Hz
    record        int                the PhysioNet record: one patient
    start         int                the window's first step, at 125 Hz in the record
    source_idx    int                the row's position in its split
    r_peaks       list[int]          each beat's R-peak, a step 0..1249 in the window
    r_peaks_360   list[int]          the same R-peak, a sample at 360 Hz in the record
    beat_class    list[int]          AAMI class: 0 N, 1 S, 2 V, 3 F, 4 Q
    symbol        list[str]          the annotation symbol (N, L, R, A, V, ...)
    rhythm        list[str]          the rhythm in effect at the beat ((N, (AFIB, ...)

The vectorizer's model_name is `raw`, which names the repo `pttrn-io/ecg-mitbih-10s-raw`.

After each build, every row is compared with the window it was cut from, and the beats
of `mitbih_beats.record_beats` are cut again from the cached windows; a mismatch stops
the script.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from datasets import Array2D, Dataset, Features, Sequence, Value
from scipy.signal import resample_poly
from vectormesh import VectorCache

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mitbih_beats
from build_heartbeat_cache import RawSequenceVectorizer
from mitbih_beats import AAMI, FS, FS_SOURCE, SPLITS

ROOT = Path(__file__).resolve().parents[1]
ARTEFACTS = ROOT / "artefacts"
MITDB = ARTEFACTS / "sources" / "mitdb"

TAG = "ecg-mitbih-10s"
STEPS = mitbih_beats.WINDOW_SECONDS * FS
INPUT_COL = "steps"
ECG_COL = "ecg"
CLASSES = ["N", "S", "V", "F", "Q"]
BEAT_LISTS = ["r_peaks", "r_peaks_360", "beat_class", "symbol", "rhythm"]

FEATURES = Features(
    {
        ECG_COL: Array2D(shape=(STEPS, 1), dtype="float32"),
        "record": Value("int64"),
        "start": Value("int64"),
        "source_idx": Value("int64"),
        "r_peaks": Sequence(Value("int64")),
        "r_peaks_360": Sequence(Value("int64")),
        "beat_class": Sequence(Value("int64")),
        "symbol": Sequence(Value("string")),
        "rhythm": Sequence(Value("string")),
    }
)


class WindowVectorizer(RawSequenceVectorizer):
    """Hand each window's steps over unchanged, as a float32 `(1250, 1)` tensor."""

    col_name: str = ECG_COL


def record_windows(source: Path, record: int) -> list[dict]:
    """The windows of one record, in time order, each a dict of the columns above
    (without `source_idx`); `steps` holds the window as float32 `(1250,)`."""
    import wfdb  # ty: ignore[import-not-found]  (not a project dependency: `uv run --with wfdb`)

    path = str(source / str(record))
    signal = wfdb.rdrecord(path)
    lead = signal.sig_name.index("MLII")
    x = resample_poly(signal.p_signal[:, lead], FS, FS_SOURCE).astype(np.float32)
    ann = wfdb.rdann(path, "atr")

    beats = []  # (peak at 125 Hz, peak at 360 Hz, class, symbol, rhythm)
    rhythm = ""
    for sample, symbol, aux in zip(ann.sample, ann.symbol, ann.aux_note, strict=True):
        if symbol == "+":
            rhythm = aux.strip("\x00 ")
        elif symbol in AAMI:
            peak = round(int(sample) * FS / FS_SOURCE)
            beats.append(
                (peak, int(sample), CLASSES.index(AAMI[symbol]), symbol, rhythm)
            )

    windows = []
    for start in range(0, len(x) - STEPS + 1, STEPS):
        inside = [b for b in beats if start <= b[0] < start + STEPS]
        windows.append(
            {
                INPUT_COL: x[start : start + STEPS],
                "record": record,
                "start": start,
                "r_peaks": [b[0] - start for b in inside],
                "r_peaks_360": [b[1] for b in inside],
                "beat_class": [b[2] for b in inside],
                "symbol": [b[3] for b in inside],
                "rhythm": [b[4] for b in inside],
            }
        )
    return windows


def split_windows(source: Path, split: str) -> list[dict]:
    """All windows of a split, record by record in the order of `SPLITS`."""
    windows = [w for r in SPLITS[split] for w in record_windows(source, r)]
    for i, w in enumerate(windows):
        w["source_idx"] = i
    return windows


def recut_beats(ecg: np.ndarray, columns: dict) -> dict[int, tuple]:
    """Cut the beats of `mitbih_beats` from cached windows of one record.

    Returns {R-peak at 360 Hz: (steps, label)}: the same rule as `record_beats`, applied
    to the cached windows and their beat lists.
    """
    out = {}
    for i, window in enumerate(ecg[:, :, 0].astype(np.float64)):
        peaks = np.asarray(columns["r_peaks"][i])
        low, high = window.min(), window.max()
        if len(peaks) < 2 or high <= low:
            continue
        scaled = np.asarray((window - low) / (high - low))
        period = float(np.median(np.diff(peaks)))
        length = min(round(mitbih_beats.SEGMENT_PERIODS * period), mitbih_beats.STEPS)
        for peak, cls, peak360 in zip(
            peaks, columns["beat_class"][i], columns["r_peaks_360"][i], strict=True
        ):
            label = CLASSES[cls]
            if label not in mitbih_beats.CLASSES or peak + length > STEPS:
                continue
            beat = np.zeros(mitbih_beats.STEPS, dtype=np.float32)
            beat[:length] = scaled[peak : peak + length]
            out[peak360] = (beat, mitbih_beats.CLASSES.index(label))
    return out


def check_cache(folder: Path, windows: list[dict], split: str) -> None:
    """Exit unless every row of the cache at `folder` equals the window it was cut
    from, and the beats of `mitbih_beats` cut from the cached windows equal its own."""
    dataset = VectorCache.load(folder).dataset
    problems = []
    if not isinstance(dataset.features[ECG_COL], Array2D):
        problems.append(f"ecg column is {dataset.features[ECG_COL]}, not Array2D")
    columns = dataset.with_format("numpy", columns=[ECG_COL])[:]
    ecg = columns[ECG_COL]
    plain = dataset.remove_columns(ECG_COL).with_format(None)[:]
    if len(ecg) != len(windows):
        problems.append(f"{len(ecg)} rows, source has {len(windows)}")
    elif plain["source_idx"] != list(range(len(windows))):
        problems.append("source_idx is not the row order")
    elif ecg.shape[1:] != (STEPS, 1) or ecg.dtype != np.float32:
        problems.append(
            f"windows are {ecg.dtype} {ecg.shape[1:]}, not float32 ({STEPS}, 1)"
        )
    else:
        source = np.stack([w[INPUT_COL] for w in windows])
        if not np.array_equal(ecg[:, :, 0], source):
            bad = np.flatnonzero((ecg[:, :, 0] != source).any(axis=1))
            problems.append(
                f"{len(bad)} windows differ from the source, first row {bad[0]}"
            )
        for col in ["record", "start", *BEAT_LISTS]:
            if plain[col] != [w[col] for w in windows]:
                problems.append(f"column {col} differs from the source")

    if not problems:
        records = np.asarray(plain["record"])
        recut_count = 0
        for record in SPLITS[split]:
            rows = np.flatnonzero(records == record)
            part = {k: [plain[k][i] for i in rows] for k in BEAT_LISTS}
            recut = recut_beats(ecg[rows], part)
            steps, labels, peaks = mitbih_beats.record_beats(MITDB, record)
            if sorted(recut) != peaks.tolist():
                problems.append(
                    f"record {record}: recut beats are not those of mitbih_beats"
                )
                continue
            for beat, label, peak in zip(steps, labels, peaks, strict=True):
                got, got_label = recut[int(peak)]
                if got_label != label or not np.allclose(got, beat, atol=1e-5):
                    problems.append(
                        f"record {record}: beat at {peak} differs when recut"
                    )
                    break
            recut_count += len(peaks)
    if problems:
        sys.exit(f"{folder.name}: " + "; ".join(problems))
    print(
        f"  checked: all {len(windows)} windows equal their source, and the "
        f"{recut_count} beats of mitbih_beats cut again from them equal its own"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--map-batch", type=int, default=256)
    args = p.parse_args()

    if not all((MITDB / f"{r}.atr").exists() for s in SPLITS.values() for r in s):
        sys.exit(
            f"{MITDB} holds no PhysioNet records; download them first:\n  make build-ecg-windows"
        )

    vectorizer = WindowVectorizer()
    for split in SPLITS:
        windows = split_windows(MITDB, split)
        dataset = Dataset.from_dict(
            {
                k: [w[k] for w in windows]
                for k in [INPUT_COL, "source_idx", "record", "start", *BEAT_LISTS]
            }
        )
        name = f"{TAG}_{split}_{vectorizer.model_name}"
        print(
            f"building {name}: {len(dataset)} windows from {len(SPLITS[split])} records"
        )
        start = time.time()
        cache = VectorCache.create(
            cache_dir=ARTEFACTS,
            vectorizer=vectorizer,
            dataset=dataset,
            dataset_tag=name,
            features=Features({**dataset.features, **FEATURES}),
            remove_columns=[INPUT_COL],
            map_batch=args.map_batch,
        )
        print(
            f"  {cache.name}  ({time.time() - start:.0f}s)\n  columns: {cache.column_names}"
        )
        check_cache(ARTEFACTS / cache.name, windows, split)

    print("\nnow check the card:\n  make publish-ecg-windows ARGS=--dry-run")


if __name__ == "__main__":
    main()
