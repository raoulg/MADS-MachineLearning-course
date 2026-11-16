"""The exercise folder of lesson 2: a design (`designs.py`) and the study that searches
it (`study.py`). Its routes, for atlas, are the design's exports."""

from .designs import Beats, Config

__all__ = ["Beats", "Config"]
