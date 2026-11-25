"""The dataset cards of the seven-domain image mix of lesson 3, one per domain and encoder.

`build_mix_cache.py build` writes them next to the caches, with the row counts of the
build; `publish_cache.py --card` uploads one.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Source:
    title: str
    hub: str  # the hub mirror the images were read from
    origin: str  # the dataset's own page
    what: str  # one sentence: what an image is
    licence: str  # the card's licence field
    terms: str  # what the origin states about use
    cite: list[str]
    splits: str  # how the cache's train and test map onto the source splits


SOURCES = {
    "dtd": Source(
        "Describable Textures Dataset (DTD)",
        "tanganke/dtd",
        "https://www.robots.ox.ac.uk/~vgg/data/dtd/",
        "a photo of a texture, labelled with one of 47 describable attributes",
        "other",
        "The origin makes the data available for research purposes.",
        [
            (
                "M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed and A. Vedaldi. Describing "
                "Textures in the Wild. CVPR 2014."
            ),
        ],
        "train and test are the mirror's `train` and `test` splits.",
    ),
    "eurosat": Source(
        "EuroSAT",
        "tanganke/eurosat",
        "https://github.com/phelber/EuroSAT",
        "a 64 x 64 Sentinel-2 RGB satellite tile, labelled with one of 10 land-use classes",
        "mit",
        "The origin publishes the data under the MIT licence; the images contain "
        "modified Copernicus Sentinel data.",
        [
            (
                "P. Helber, B. Bischke, A. Dengel and D. Borth. EuroSAT: A Novel Dataset and "
                "Deep Learning Benchmark for Land Use and Land Cover Classification. IEEE "
                "Journal of Selected Topics in Applied Earth Observations and Remote Sensing, "
                "2019."
            ),
        ],
        "train and test are the mirror's `train` and `test` splits; its corrupted "
        "copies of the test split are left out.",
    ),
    "oxford-flowers": Source(
        "Oxford 102 Flowers",
        "nkirschi/oxford-flowers",
        "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/",
        "a photo of a flower, labelled with one of 102 species; the class names are "
        "the category numbers of the origin, as the mirror stores them",
        "unknown",
        "The origin states no licence.",
        [
            (
                "M.-E. Nilsback and A. Zisserman. Automated Flower Classification over a Large "
                "Number of Classes. ICVGIP 2008."
            ),
        ],
        "train and test are the mirror's `train` and `test` splits.",
    ),
    "food101": Source(
        "Food-101",
        "ethz/food101",
        "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/",
        "a photo of a dish, labelled with one of 101 food categories",
        "other",
        "The images come from foodspotting.com; the origin does not own them and "
        "provides them for research under scientific fair use.",
        [
            (
                "L. Bossard, M. Guillaumin and L. Van Gool. Food-101 -- Mining Discriminative "
                "Components with Random Forests. ECCV 2014."
            ),
        ],
        "train is the source's `train` split; test is its `validation` split, the "
        "official test set.",
    ),
    "gtsrb": Source(
        "German Traffic Sign Recognition Benchmark (GTSRB)",
        "tanganke/gtsrb",
        "https://benchmark.ini.rub.de/gtsrb_dataset.html",
        "a photo of a traffic sign, labelled with one of 43 sign classes",
        "unknown",
        "The origin states no licence and asks that the benchmark paper be cited.",
        [
            (
                "J. Stallkamp, M. Schlipsing, J. Salmen and C. Igel. Man vs. computer: "
                "Benchmarking machine learning algorithms for traffic sign recognition. "
                "Neural Networks 32, 2012."
            ),
        ],
        "train and test are the mirror's `train` and `test` splits; its corrupted "
        "copies of the test split are left out.",
    ),
    "galaxy10-decals": Source(
        "Galaxy10 DECaLS",
        "matthieulel/galaxy10_decals",
        "https://astronn.readthedocs.io/en/latest/galaxy10.html",
        "a 256 x 256 colour image of a galaxy from the DESI Legacy Imaging Surveys, "
        "labelled with one of 10 morphology classes from Galaxy Zoo votes",
        "cc-by-4.0",
        "The origin publishes Galaxy10 DECaLS under CC BY 4.0.",
        [
            "H. W. Leung and J. Bovy. Galaxy10 DECaLS, astroNN (2021).",
            (
                "M. Walmsley et al. Galaxy Zoo DECaLS: Detailed visual morphology "
                "measurements from volunteers and deep learning for 314 000 galaxies. "
                "MNRAS 509, 2022."
            ),
            "A. Dey et al. Overview of the DESI Legacy Imaging Surveys. AJ 157, 2019.",
        ],
        "train and test are the mirror's `train` and `test` splits.",
    ),
    "bloodmnist": Source(
        "BloodMNIST (MedMNIST v2)",
        "albertvillanova/medmnist-v2",
        "https://medmnist.com/",
        "a 28 x 28 microscope image of a blood cell, labelled with one of 8 cell types",
        "cc-by-4.0",
        "The origin publishes BloodMNIST under CC BY 4.0.",
        [
            (
                "J. Yang et al. MedMNIST v2 -- A large-scale lightweight benchmark for 2D and "
                "3D biomedical image classification. Scientific Data 10, 2023."
            ),
            (
                "A. Acevedo et al. A dataset of microscopic peripheral blood cell images for "
                "development of automatic recognition systems. Data in Brief 30, 2020."
            ),
        ],
        "train is the source's `train` and `validation` splits together (the "
        "`source_split` column says which); test is its `test` split. The images were "
        "read from the `bloodmnist` config of the mirror's parquet conversion.",
    ),
}


def dots(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def write_card(
    domain: str,
    encoder: str,
    dim: int,
    names: list[str],
    counts: dict[str, tuple[int, np.ndarray]],
) -> str:
    s = SOURCES[domain]
    slug = encoder.split("/")[-1]
    repo = f"pttrn-io/{domain}-cls-mean-{slug}"
    n_train, per_train = counts["train"]
    n_test, _ = counts["test"]
    cite = "\n".join(f"- {c}" for c in s.cite)
    classes = ", ".join(f"`{n}`" for n in names[:6]) + (
        ", ..." if len(names) > 6 else ""
    )
    return f"""---
license: {s.licence}
tags:
  - vectormesh
  - vector-cache
  - embeddings
  - image-classification
---

# {s.title}: `{slug}` CLS and patch mean

Every image of {s.title} as two `{slug}` vectors, so a head trains on a laptop
without running the encoder. One of the seven domains of the image mix of lesson 3 of
the [MADS MachineLearning course](https://github.com/raoulg/MADS-MachineLearning-course).

## What a row is

One image: {s.what}.

| column | type | |
|---|---|---|
| `embed` | float16 `(2, {dim})`, a fixed-shape `Array2D` | row 0 the CLS token, row 1 the mean of the patch tokens |
| `label` | int, a `ClassLabel` of {len(names)} classes | {classes} |
| `source_idx` | int | the image's position in its source split |
| `source_split` | string | the source split the image comes from |

Both vectors come from the encoder's last layer after its final layer norm. The CLS
token is what `vectormesh.ImageVectorizer` pools a DINOv2 to; the patch mean averages
the 256 patch tokens of the 224 x 224 crop. They are stored in float16; reading the
cache gives float32. A head trained on the float16 vectors scores as well as on the
float32 vectors they were rounded from, within what another seed changes (DTD and
EuroSAT, `dinov2-small`, three seeds).

| | |
|---|---|
| encoder | `{encoder}` |
| hidden_size | {dim} |
| rows | {dots(n_train)} train / {dots(n_test)} test |
| train rows per class | {int(per_train.min())} to {int(per_train.max())} |
| source | `{s.hub}` on the hub, from <{s.origin}> |
| licence | {s.licence} |

Splits: {s.splits}

## Load it

```python
from vectormesh import VectorCache

train = VectorCache.from_hub("{repo}", split="train")
test = VectorCache.from_hub("{repo}", split="test")
X = train.dataset.with_format("torch")["embed"]  # (rows, 2, {dim})
cls, patch_mean = X[:, 0], X[:, 1]
```

Pin `revision=` to a commit for anything graded. This is a `save_to_disk` layout, not
parquet: `load_dataset()` will not read it, `VectorCache.load()` and
`VectorCache.from_hub()` will. There is no validation split: carve one from train and
leave test alone until the end.

## How it was built

`scripts/build_mix_cache.py` in
<https://github.com/raoulg/MADS-MachineLearning-course> reads the parquet files of
`{s.hub}`, runs every image through `{encoder}` and stores the two vectors with the
label and the source position.

## Licence and attribution

{s.terms} This repository holds vectors computed from the images, not the images. Use it
under the terms of the source and cite:

{cite}
"""
