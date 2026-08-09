"""SCOTUS specifics shared by this lesson's build scripts and notebooks.

The cache holds vectors; the LLM branch of the router needs the opinion *text*. Rows carry
`source_split` and `source_idx` so one can be turned back into the other, and `opinions()`
does that lookup — with the same truncation the encoder used, so both branches of the
cascade are judging the same document.
"""

from __future__ import annotations

from datasets import load_dataset

SOURCE = "coastalcph/lex_glue"
CONFIG = "scotus"
CACHE_REPO = "pttrn-io/scotus-bge-large-en-v1.5"

# A property of the published cache, not a knob: change it and the vectors no longer
# describe the text they were built from.
DOC_CHARS = 24_000

# The generator gets a quarter of that. On 40 stratified test opinions, 6k and 24k scored
# the same 0.650 while 24k took 3.6x as long (4.7 vs 1.3 s/doc) -- the issue area is settled
# by the opening of an opinion, and the rest is argument. The two branches of the cascade
# therefore read different amounts of the same document, each the amount it was measured to
# want, which is the choice an engineer makes rather than the symmetric-looking one.
LLM_CHARS = 6_000

# The loaded dataset exposes 13 classes named '1'..'13'; the LexGLUE card describes 14 issue
# areas (the 14th, "Private Action", is absent here). Index i -> issue area i+1.
AREAS = (
    "Criminal Procedure",
    "Civil Rights",
    "First Amendment",
    "Due Process",
    "Privacy",
    "Attorneys",
    "Unions",
    "Economic Activity",
    "Judicial Power",
    "Federalism",
    "Interstate Relations",
    "Federal Taxation",
    "Miscellaneous",
)


def opinions(dataset, doc_chars: int = DOC_CHARS) -> list[str]:
    """The opinion behind each row of a SCOTUS cache, in cache order.

    `dataset` is the `.dataset` of a loaded VectorCache. Each source split is loaded once,
    however the rows are interleaved.
    """
    splits = {
        name: load_dataset(SOURCE, CONFIG, split=name)
        for name in sorted(set(dataset["source_split"]))
    }
    return [
        splits[split][int(idx)]["text"][:doc_chars]
        for split, idx in zip(dataset["source_split"], dataset["source_idx"])
    ]
