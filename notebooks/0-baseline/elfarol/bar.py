from typing import List

import seaborn as sns

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
