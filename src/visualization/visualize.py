from typing import Dict

import seaborn as sns


def plot_timers(timer: Dict[str, float]) -> None:
    x = list(timer.keys())
    y = list(timer.values())
    sns.barplot(x=x, y=y)
