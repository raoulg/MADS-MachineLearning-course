"""The exercise folder of lesson 3: a design (`designs.py`) and the study that searches
it (`study.py`). Its routes, for atlas, are the design's exports."""

import logging

# on macOS, torch warns about redirects of its distributed launcher, which nothing here
# uses
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(
    logging.ERROR
)

from .designs import Config, Head, Mix  # ty: ignore[import-not-found]  (the folder is no package on ty's search path)

__all__ = ["Config", "Head", "Mix"]
