"""The exercise folder of lesson 1: a design (`designs.py`) and the study that searches
it (`study.py`). Its routes, for atlas, are the design's exports."""

from .designs import Config, Head  # ty: ignore[import-not-found]  (the folder is no package on ty's search path)

__all__ = ["Config", "Head"]
