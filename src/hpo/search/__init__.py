"""What a study searches: the space it draws configs from, and the budget of trials the
student's minutes buy. The design whose options those configs switch is
`vectormesh.design.Design`."""

from hpo.search.budget import Budget
from hpo.search.space import Axis, Choice, Config, Float, Int, Param, Space

__all__ = [
    "Axis",
    "Budget",
    "Choice",
    "Config",
    "Float",
    "Int",
    "Param",
    "Space",
]
