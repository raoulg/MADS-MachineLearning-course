"""Build the VectorCaches of the seven-domain image mix of lesson 3, one per encoder.

    uv run scripts/build_mix_cache.py embed --encoder facebook/dinov2-small
    uv run scripts/build_mix_cache.py build --encoder facebook/dinov2-small
    uv run scripts/build_mix_cache.py embed --encoder facebook/dinov2-small --limit 512   # a timing run

Seven image classification datasets, every image of each, in their own train and test
split. `embed` reads the parquet files of each source from the hub, row group by row
group, and stores the encoder's pooled vector of every image (the layer-normed CLS token,
as `vectormesh.ImageVectorizer` pools a DINOv2) in float32 chunks under
`artefacts/sources/mix7/<encoder>/`; a chunk that exists is skipped, so an interrupted
run resumes. `build` joins the chunks into two VectorCache folders, train and test, with
the vectors in float16, ready for `scripts/publish_cache.py`.

Columns: `embed` (float16), `label` (one class of the whole mix, its name
`<domain>/<class>`), `domain` (the dataset), `domain_label` (the class within its
dataset) and `source_idx` (the row's position in its source split).

Splits: each source's own train and test split. Food-101 calls its test split
`validation`; BloodMNIST's validation split joins its train split. The GTSRB and EuroSAT
mirrors also carry corrupted copies of the test split, which are left out.
"""

import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple, Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from beartype import beartype
from datasets import ClassLabel, Dataset, Features, Sequence, Value
from huggingface_hub import HfApi, hf_hub_download
from jaxtyping import Float, jaxtyped
from PIL import Image
from pydantic import model_validator
from torch import Tensor
from vectormesh import VectorCache
from vectormesh.data.vectorizers import ChunkedVectorizer

ROOT = Path(__file__).resolve().parents[1]
ARTEFACTS = ROOT / "artefacts"
CHUNKS = ARTEFACTS / "sources" / "mix7"
MEDMNIST_PARQUET = "refs/convert/parquet"

GALAXY10 = [
    "disturbed", "merging", "round_smooth", "in_between_round_smooth",
    "cigar_shaped_smooth", "barred_spiral", "unbarred_tight_spiral",
    "unbarred_loose_spiral", "edge_on_without_bulge", "edge_on_with_bulge",
]  # fmt: skip

# domain -> (hub repo, revision, {split: parquet file prefixes in the repo})
SOURCES: dict[str, tuple[str, str, dict[str, list[str]]]] = {
    "dtd": ("tanganke/dtd", "main", {"train": ["data/train-"], "test": ["data/test-"]}),
    "eurosat": (
        "tanganke/eurosat", "main", {"train": ["data/train-"], "test": ["data/test-"]}
    ),
    "oxford-flowers": (
        "nkirschi/oxford-flowers", "main",
        {"train": ["data/train-"], "test": ["data/test-"]},
    ),
    "food101": (
        "ethz/food101", "main", {"train": ["data/train-"], "test": ["data/validation-"]}
    ),
    "gtsrb": ("tanganke/gtsrb", "main", {"train": ["data/train-"], "test": ["data/test-"]}),
    "galaxy10-decals": (
        "matthieulel/galaxy10_decals", "main",
        {"train": ["data/train-"], "test": ["data/test-"]},
    ),
    "bloodmnist": (
        "albertvillanova/medmnist-v2", MEDMNIST_PARQUET,
        {"train": ["bloodmnist/train/", "bloodmnist/validation/"],
         "test": ["bloodmnist/test/"]},
    ),
}  # fmt: skip


def parquet_files(repo: str, revision: str, prefix: str) -> list[str]:
    files = HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))


def card_names(repo: str) -> list[str] | None:
    """The label's class names in the `dataset_info` of a source's dataset card."""
    path = hf_hub_download(repo, "README.md", repo_type="dataset")
    text = Path(path).read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    info = yaml.safe_load(text.split("---")[1]).get("dataset_info", {})
    for config in info if isinstance(info, list) else [info]:
        for feature in config.get("features", []):
            if feature.get("name") == "label" and isinstance(
                feature.get("dtype"), dict
            ):
                names = feature["dtype"].get("class_label", {}).get("names")
                if names:
                    return [names[k] for k in sorted(names, key=int)]
    return None


def class_names(repo: str, revision: str, files: list[str], domain: str) -> list[str]:
    """The class names a source carries (a ClassLabel in its parquet schema or on its
    dataset card), else known ones."""
    if domain == "galaxy10-decals":
        return GALAXY10
    path = hf_hub_download(repo, files[0], repo_type="dataset", revision=revision)
    meta = pq.read_schema(path).metadata or {}
    info = json.loads(meta.get(b"huggingface", b"{}").decode("utf-8"))
    label = info.get("info", {}).get("features", {}).get("label", {})
    names = label.get("names") or card_names(repo)
    if not names:
        raise SystemExit(f"{repo}: no class names in the parquet schema or the card")
    return names


def decode(item: dict) -> Image.Image:
    return Image.open(io.BytesIO(item["bytes"])).convert("RGB")


def embed(encoder: str, limit: int | None, batch: int) -> None:
    from vectormesh import ImageVectorizer

    vectorizer = ImageVectorizer(model_name=encoder)
    out = CHUNKS / encoder.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    print(f"{encoder}: dim {vectorizer.get_hidden_size}, device {vectorizer.device}")
    pool = ThreadPoolExecutor(max_workers=6)
    total, start = 0, time.time()
    for domain, (repo, revision, splits) in SOURCES.items():
        names: list[str] = []
        for split, prefixes in splits.items():
            offset = 0
            for prefix in prefixes:
                files = parquet_files(repo, revision, prefix)
                names = names or class_names(repo, revision, files, domain)
                tag = prefix.strip("/-").replace("/", "-")
                for k, name in enumerate(files):
                    path = hf_hub_download(
                        repo, name, repo_type="dataset", revision=revision
                    )
                    f = pq.ParquetFile(path)
                    for g in range(f.num_row_groups):
                        n = f.metadata.row_group(g).num_rows
                        part = out / f"{domain}_{split}_{tag}_{k:02d}_{g:03d}.npz"
                        rows = np.arange(offset, offset + n)
                        offset += n
                        if part.exists():
                            continue
                        table = f.read_row_group(g, columns=["image", "label"])
                        labels = np.asarray(table.column("label").to_pylist())
                        labels = labels.reshape(n, -1)[:, 0].astype(np.int64)
                        images = table.column("image").to_pylist()
                        if limit:
                            images, labels = images[:limit], labels[:limit]
                            rows = rows[:limit]
                        tick = time.time()
                        pil = list(pool.map(decode, images))
                        vectors = vectorizer(pil, batch)[vectorizer.col_name]
                        X = torch.stack(vectors).float().cpu().numpy()
                        if limit:
                            rate = len(pil) / (time.time() - tick)
                            print(
                                f"timing: {len(pil)} images of {domain}, {rate:.0f}/s"
                            )
                            return
                        np.savez(part, X=X, y=labels, row=rows)
                        total += len(pil)
                        rate = total / (time.time() - start)
                        print(
                            f"{part.name}: {len(pil)} images, {total} in all, {rate:.0f}/s",
                            flush=True,
                        )
        names_file = out / f"{domain}.classes.json"
        names_file.write_text(json.dumps(names), encoding="utf-8")
    print(f"embedded {total} images in {time.time() - start:.0f}s")


class StoredVectorizer(ChunkedVectorizer):
    """Hand each row's stored vector over as a float16 tensor: the encoder ran in
    `embed`, so this one only fixes the precision the cache keeps."""

    model_name: str = "dinov2-small"
    col_name: str = "embed"
    input_col: str = "vector"
    dim: int = 384

    @model_validator(mode="after")
    def initialize_model(self) -> Self:
        dim = self.dim

        class StoredMetadata:
            hidden_size = dim

        self._metadata = StoredMetadata()
        self._effective_max_length = None
        return self

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, inputs: list, batchsize: int
    ) -> dict[str, list[Float[Tensor, " dim"]]]:
        batch = torch.tensor(np.asarray(inputs, dtype=np.float32)).half()
        return {self.col_name: list(batch)}


class Rows(NamedTuple):
    X: np.ndarray
    label: np.ndarray
    domain: np.ndarray
    domain_label: np.ndarray
    row: np.ndarray
    names: list[str]


def read_chunks(folder: Path, split: str) -> Rows:
    """Every embedded row of `split`, domain by domain in `SOURCES` order, with the
    labels of the whole mix and the class names `<domain>/<class>`."""
    X, label, domain, domain_label, row, names = [], [], [], [], [], []
    for d, domain_name in enumerate(SOURCES):
        classes = json.loads(
            (folder / f"{domain_name}.classes.json").read_text(encoding="utf-8")
        )
        parts = sorted(folder.glob(f"{domain_name}_{split}_*.npz"))
        if not parts:
            raise SystemExit(
                f"{folder}: no {split} chunks of {domain_name}; embed first"
            )
        offset = len(names)
        for part in parts:
            z = np.load(part)
            X.append(z["X"])
            label.append(z["y"] + offset)
            domain_label.append(z["y"])
            domain.append(np.full(len(z["y"]), d))
            row.append(z["row"])
        names += [f"{domain_name}/{c}" for c in classes]
    joined = (np.concatenate(a) for a in (X, label, domain, domain_label, row))
    return Rows(*joined, names=names)


def build(encoder: str) -> None:
    slug = encoder.split("/")[-1]
    folder = CHUNKS / slug
    for split in ("train", "test"):
        data = read_chunks(folder, split)
        X = data.X
        dataset = Dataset.from_dict(
            {
                "vector": list(X),
                "label": data.label.tolist(),
                "domain": data.domain.tolist(),
                "domain_label": data.domain_label.tolist(),
                "source_idx": data.row.tolist(),
            },
            features=Features(
                {
                    "vector": Sequence(Value("float32")),
                    "label": ClassLabel(names=data.names),
                    "domain": ClassLabel(names=list(SOURCES)),
                    "domain_label": Value("int64"),
                    "source_idx": Value("int64"),
                }
            ),
        )
        vectorizer = StoredVectorizer(model_name=slug, dim=X.shape[1])
        name = f"mix7_{split}_{slug}"
        print(f"building {name}: {len(dataset)} rows, {X.shape[1]} dims")
        cache = VectorCache.create(
            cache_dir=ARTEFACTS,
            vectorizer=vectorizer,
            dataset=dataset,
            dataset_tag=name,
            features=Features(
                {
                    **{k: v for k, v in dataset.features.items() if k != "vector"},
                    "embed": Sequence(Value("float16")),
                }
            ),
            remove_columns=["vector"],
            map_batch=4096,
        )
        built = VectorCache.load(ARTEFACTS / cache.name).dataset
        stored = built.data.schema.field("embed").type.value_type
        assert stored == pa.float16(), stored  # reading upcasts it to float32
        built = built.with_format("numpy")
        assert (np.asarray(built["label"]) == data.label).all()
        error = np.abs(np.asarray(built["embed"]) - X).max()
        size = sum(f.stat().st_size for f in (ARTEFACTS / cache.name).rglob("*"))
        print(f"  {cache.name}: {size / 1e6:.0f} MB, largest fp16 rounding {error:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("step", choices=["embed", "build"])
    p.add_argument("--encoder", required=True)
    p.add_argument("--limit", type=int, help="embed only this many images (timing)")
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()
    if args.step == "embed":
        embed(args.encoder, args.limit, args.batch)
    else:
        build(args.encoder)


if __name__ == "__main__":
    main()
