import numpy as np

from hpo.measure.compare import Comparison


def test_one_run_prints_no_spread():
    line = str(Comparison.of("baseline", np.array([0.8676])))
    assert "+/-" not in line
    assert "(n=1)" in line


def test_repeated_runs_print_their_spread():
    line = str(Comparison.of("winner", np.array([0.85, 0.86, 0.87])))
    assert "+/- 0.0082" in line
