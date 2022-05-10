from typing import Callable, List, Sequence, Union

import numpy as np
from loguru import logger
from scipy import stats


class Hypotheses:
    def __init__(self, n: Union[int, List[int]], fixed: bool = False):
        """Provide one or more functions that will take in a history,
        and output a prediction

        Args:
            n (Union[int, List[int]]): if fixed is False, this should be an int
            if fixed is true, this can should be a list of integers.

            If not fixed, n is the number of random hypotheses picked
            If fixed, the numbers are the index that selects one of the available
            functions.

            fixed (bool, optional): [description]. Defaults to False.

        Raises:
            TypeError: [description]
        """
        self._all_h: List[Callable[[List[int]], float]] = [
            self._mirror,
            self._average_4w,
            self._cycle2w,
            self._cycle4w,
            self._cycle5w,
            self._trend,
            self._random,
            self._lastweek,
        ]

        if not fixed:
            if isinstance(n, int) and n > len(self._all_h):
                logger.info(f"Max value of n is {len(self._all_h)}, found {n}")
                n = len(self._all_h)

            if isinstance(n, int) and n > 0:
                idx = list(np.random.choice(np.arange(0, len(self._all_h)), size=n, replace=False))  # type: ignore
            elif isinstance(n, int) and n == 0:
                idx = [-1]
            else:
                raise TypeError(
                    f"If fixed is False, n should be and int, found {type(n)}"
                )
        else:
            if isinstance(n, list) and np.max(n) >= len(self._all_h):
                raise ValueError(f"Item in n too big: {np.max(n)}")
            if isinstance(n, int):
                raise TypeError(
                    f"If fixed is True, n should be a list of integers, found int"
                )
            idx = n

        self.models: List[Callable[[List[int]], float]] = [self._all_h[i] for i in idx]
        self.n = n

    def __len__(self):
        return len(self.models)

    def __repr__(self):
        names = [x.__name__[1:] for i, x in enumerate(self.models)]
        return f"Hypotheses(models={names})"

    def _random(self, hist: List[int]) -> float:
        return float(np.random.choice(hist))

    def _mirror(self, hist: List[int]) -> float:
        lw = hist[-1]
        return float((50 - lw) + 50)

    def _average_4w(self, hist: List[int]) -> float:
        if len(hist) >= 4:
            lw = hist[-4:]
        else:
            lw = hist
        return float(np.mean(lw))

    def _fallback(self, hist: List[int], k: int) -> float:
        if len(hist) >= k:
            x = hist[-k]
        else:
            x = hist[0]
        return float(x)

    def _lastweek(self, hist: List[int]) -> float:
        x = self._fallback(hist, k=1)
        return x

    def _cycle2w(self, hist: List[int]) -> float:
        x = self._fallback(hist, k=2)
        return x

    def _cycle4w(self, hist: List[int]) -> float:
        x = self._fallback(hist, k=4)
        return x

    def _cycle5w(self, hist: List[int]) -> float:
        x = self._fallback(hist, k=5)
        return x

    def _trend(self, hist: List[int], n: int = 8) -> float:
        if len(hist) == 1:
            yhat = self._fallback(hist, k=1)
        else:
            y = hist[-n:]
            x = np.arange(len(y))
            model = stats.linregress(x, y)
            yhat = model.slope * len(y) + model.intercept
        return yhat
