from typing import Callable, List, Union

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
        # all available hypotheses
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
            # if n is an integer
            if isinstance(n, int) and n > len(self._all_h):
                # pick all available hypotheses
                logger.info(f"Max value of n is {len(self._all_h)}, found {n}")
                n = len(self._all_h)

            if isinstance(n, int) and n > 0:
                # pick n random hypotheses
                idx = list(
                    np.random.choice(
                        np.arange(0, len(self._all_h)), size=n, replace=False
                    )
                )  # type: ignore
            elif isinstance(n, int) and n == 0:
                # if 0, pick the last one
                idx = [-1]
            else:
                raise TypeError(
                    f"If fixed is False, n should be and int, found {type(n)}"
                )
        else:
            # if n is a list and fixed=True, the list defines which hypotheses are
            # picked and in what order
            if isinstance(n, list) and np.max(n) >= len(self._all_h):
                raise ValueError(f"Item in n too big: {np.max(n)}")
            if isinstance(n, int):
                raise TypeError(
                    "If fixed is True, n should be a list of integers, found int"
                )
            idx = n

        # models is a list of callable hypotheses
        self.models: List[Callable[[List[int]], float]] = [self._all_h[i] for i in idx]
        self.n = n

    def __len__(self):
        return len(self.models)

    def __repr__(self):
        names = [x.__name__[1:] for i, x in enumerate(self.models)]
        return f"Hypotheses(models={names})"

    def _random(self, hist: List[int]) -> float:
        # pick a random value from the past
        return float(np.random.choice(hist))

    def _mirror(self, hist: List[int]) -> float:
        # pick the previous number, and mirror around 50
        lw = hist[-1]
        return float((50 - lw) + 50)

    def _average_4w(self, hist: List[int]) -> float:
        # pick the average of the last four weeks
        if len(hist) >= 4:
            lw = hist[-4:]
        else:
            lw = hist
        return float(np.mean(lw))

    def _fallback(self, hist: List[int], k: int) -> float:
        # pick the last k=th item
        if len(hist) >= k:
            x = hist[-k]
        else:
            x = hist[0]
        return float(x)

    def _lastweek(self, hist: List[int]) -> float:
        # pick the last item
        x = self._fallback(hist, k=1)
        return x

    def _cycle2w(self, hist: List[int]) -> float:
        # pick the item from two weeks ago
        x = self._fallback(hist, k=2)
        return x

    def _cycle4w(self, hist: List[int]) -> float:
        # pick the number 4 weeks ago
        x = self._fallback(hist, k=4)
        return x

    def _cycle5w(self, hist: List[int]) -> float:
        # pick the number 5 weeks ago
        x = self._fallback(hist, k=5)
        return x

    def _trend(self, hist: List[int], n: int = 8) -> float:
        # calculate a linear regression, and predict this as the next amount
        # you are now allowed to put into your powerpoint salespitch that your
        # product uses "ai" and "algorithms" :'D
        if len(hist) == 1:
            yhat = self._fallback(hist, k=1)
        else:
            y = hist[-n:]
            x = np.arange(len(y))
            model = stats.linregress(x, y)
            yhat = model.slope * len(y) + model.intercept
        return yhat
