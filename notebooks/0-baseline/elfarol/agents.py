from typing import Callable, List, Optional, Sequence, Union

import numpy as np

import elfarol.hypotheses as hypt


class BaseAgent:
    def __init__(self, threshold: int) -> None:
        """This generates a basic agent with a set of Hypotheses and a threshold

        Args:
            threshold (int): Every agent has a threshold of what they consider 
            a comfortable amount of people in the bar. By default this is fixated 
            at the 7th hypothesis
        

        """
        self.threshold = self._get_threshold(threshold)
        self.hypotheses: hypt.Hypotheses = hypt.Hypotheses(n=[7], fixed=True)
        self.log: List[str] = []

    def _get_threshold(self, threshold: int, hist: Optional[List[int]] = None) -> int:
        return threshold

    def __repr__(self):
        return f"Agent(threshold={self.threshold}, model={self.hypotheses})"

    def _predict(self, hist: List[int]) -> float:
        """Run some sort of prediction, based on the history,
        for the amount of people that will come

        Args:
            hist (List[int]): historic list of people that came last week to the
            bar

        Returns:
            float: prediction for upcoming week
        """
        model = self._getmodel(hist)
        self.log.append(str(model.__name__))
        yhat = model(hist)
        return yhat

    def _getmodel(self, hist: List[int]) -> Callable[[List[int]], float]:
        return self.hypotheses.models[0]

    def decide(self, hist: List[int]) -> bool:
        """The agent makes a prediction for upcoming week, based on the history,
        and decides if he will go (True) or stay home (False)

        Args:
            hist (List[int]): historic list of people that came last week to the
            bar

        Returns:
            bool: Will the agent go, or stay home?
        """
        yhat = self._predict(hist)
        return yhat <= self.threshold


class Agent(BaseAgent):
    def __init__(self, n: Union[int, List[int]], threshold: int = 60, fixed: bool = False) -> None:
        """By default, the agent has a threshold of 60 and generates a random 
        order of n hypotheses at initialisation.

        Args:
            n (int or list[int]): number of hypotheses to use
            threshold (int, optional): Defaults to 60.
            fixed (bool, optional):  If the order of hypotheses should be fixed. If 
            True, n should be a list. Defaults to False.
        """
        super().__init__(threshold=threshold)
        self.hypotheses: hypt.Hypotheses = hypt.Hypotheses(n, fixed=fixed)

    def _getmodel(self, hist: List[int]) -> Callable[[List[int]], float]:
        if len(hist) == 1:
            past = hist
        else:
            past = hist[:-1]

        yhat = []
        for model in self.hypotheses.models:
            yhat.append(model(past))

        diff = np.abs(np.array(yhat) - hist[-1])
        idx = np.argmin(diff)
        return self.hypotheses.models[idx]

