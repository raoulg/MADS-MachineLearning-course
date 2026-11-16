from collections import Counter
from collections.abc import Sequence

import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from loguru import logger

import elfarol.hypotheses as hypt
from elfarol.agents import Agent, BaseAgent


class MoodyAgent(Agent):
    def __init__(
        self, n: int | list[int], threshold: int = 60, fixed: bool = False
    ) -> None:
        super().__init__(n=n, threshold=threshold, fixed=fixed)
        self.hypotheses: hypt.Hypotheses = hypt.Hypotheses(n, fixed=fixed)

    def _get_threshold(self, threshold: int, hist: list[int] | None = None) -> int:
        noise = np.random.randint(-9, 10)

        return threshold + noise

    def _predict(self, hist: list[int]) -> float:
        """Run some sort of prediction, based on the history,
        for the amount of people that will come

        Args:
            hist (List[int]): historic list of people that came last week to the
            bar

        Returns:
            float: prediction for upcoming week
        """
        model = self._getmodel(hist)
        self.log.append(str(getattr(model, "__name__", str(model))))
        yhat = model(hist)
        return yhat

    def decide(self, hist: list[int]) -> bool:
        yhat = self._predict(hist)
        return yhat <= self._get_threshold(self.threshold)


class BaseExperiment:
    def __init__(self, agents: Sequence[BaseAgent], hist: list[int] = [100]) -> None:
        self.agents = agents
        self.hist = hist

    def saturdaynight(self) -> list[int]:
        k = 0
        for agent in self.agents:
            if agent.decide(self.hist):
                k += 1
        self.hist.append(k)
        return self.hist

    def simulate(self, k: int = 52) -> list[int]:
        for _ in range(k):
            self.saturdaynight()
        return self.hist


class Experiment(BaseExperiment):
    def __init__(self, agents: Sequence[BaseAgent], hist: list[int] = [100]) -> None:
        super().__init__(agents=agents, hist=hist)

    def line(self):
        sns.lineplot(x=range(len(self.hist)), y=self.hist)

    def hist_n_hypotheses(self):
        n = []
        for agent in self.agents:
            n.append(len(agent.hypotheses))
        sns.histplot(n)

    def bar_hypt_population(self, figsize: tuple[float, float] = (10, 10)):
        types = []
        for agent in self.agents:
            for model in agent.hypotheses.models:
                types.append(getattr(model, "__name__", str(model)))
        c = Counter(types)
        plt.figure(figsize=figsize)
        sns.barplot(x=list(c.keys()), y=list(c.values()))
        plt.xticks(rotation=90)

    def bar_log(self, figsize: tuple[float, float] = (10, 10)):
        models_used = [name for x in self.agents for name in x.log]
        normalize = len(models_used)
        logger.info(f"Found {normalize} models.")

        count = Counter(models_used)
        sortcount = {k: v for k, v in count.most_common()}
        y = np.array(list(sortcount.values())) / normalize

        plt.figure(figsize=figsize)
        sns.barplot(x=list(sortcount.keys()), y=y)
