from collections import Counter
from typing import List, Tuple

import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from loguru import logger

from elfarol.agents import BaseAgent


def saturdaynight(agents: List[BaseAgent], hist: List[int]) -> List[int]:
    k = 0
    for agent in agents:
        if agent.decide(hist):
            k += 1
    hist.append(k)
    return hist


def simulate(
    agents: List[BaseAgent], first_night: int = 100, k: int = 52, plot: bool = True
) -> List[int]:
    hist = [first_night]
    for _ in range(k):
        saturdaynight(agents, hist)
    if plot:
        x = range(len(hist))
        sns.lineplot(x=x, y=hist)
    return hist


class BaseExperiment:
    def __init__(self, agents: List[BaseAgent], hist: List[int] = [100]) -> None:
        self.agents = agents
        self.hist = hist

    def saturdaynight(self) -> List[int]:
        k = 0
        for agent in self.agents:
            if agent.decide(self.hist):
                k += 1
        self.hist.append(k)
        return self.hist

    def simulate(self, k: int = 52) -> List[int]:
        for _ in range(k):
            self.saturdaynight()
        return self.hist


class Experiment(BaseExperiment):
    def __init__(self, agents: List[BaseAgent], hist: List[int] = [100]) -> None:
        super().__init__(agents=agents, hist=hist)

    def line(self):
        sns.lineplot(x=range(len(self.hist)), y=self.hist)

    def hist_n_hypotheses(self):
        n = []
        for agent in self.agents:
            n.append(len(agent.hypotheses))
        sns.histplot(n)

    def bar_hypt_population(self, figsize: Tuple = (10, 10)):
        types = []
        for agent in self.agents:
            for model in agent.hypotheses.models:
                types.append(model.__name__)
        c = Counter(types)
        plt.figure(figsize=figsize)
        sns.barplot(x=list(c.keys()), y=list(c.values()))
        plt.xticks(rotation=90)

    def bar_log(self, figsize: Tuple = (10, 10)):
        models_used = [name for x in self.agents for name in x.log]
        normalize = len(models_used)
        logger.info(f"Found {normalize} models.")

        count = Counter(models_used)
        sortcount = {k: v for k, v in count.most_common()}
        y = np.array(list(sortcount.values())) / normalize

        plt.figure(figsize=figsize)
        sns.barplot(x=list(sortcount.keys()), y=y)
