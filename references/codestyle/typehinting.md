# Typehinting

## Basic typehinting

Since python version 3.5, there is support for typehinting.
An overview of all PEPs can be found [here](https://docs.python.org/3/library/typing.html)
A simplified introduction can be found in [PEP 483](https://www.python.org/dev/peps/pep-0483/) , While
a full specification can be found in [PEP 484](https://www.python.org/dev/peps/pep-0484/)

As a very short introduction: instead of code like this

```python
def impute(
    df, group_cols, metric = "mean"
):
...
```

```python
from typing import List
import pandas as pd


def impute(
    df: pd.DataFrame, group_cols: List[str], metric: str = "mean"
) -> pd.DataFrame:
    ...
```

While it often seems clear at the time you wrote the code what every argument
should look like, and often can be inferred from e.g. a notebook or some other context,
I might not be that clear to someone else that needs to read your code, or even to yourselve after 6 months.

In addition to that, your code might still run if you pass a wrong type, causing unexpected errors later on. If you start using `mypy` as a linter, it will check your typehints for consistency.

- The most common types are things like `str`, `int`, `float`, `List`, `Set`, `Dict`.
- There are additional constructs like `None`, `Union`, `Tuple` and `Optional`.

## Custom types

At some point your types might become complex. A typical example is a "sentence" when doing NLP.
A sentence could be a list of words, so it might look like this:

```python
sentence = ["This", "is", "a", "sentence"]
```

This will be typehinted as `List[str]`.  However, your sentence might still be a `List[str]` but it could look like this:

```python
sentence = ["This is a sentence"]
```

This gets confusing fast. In addition to that, you might get sentences that are cleaned (eg html tags are removed, punctuation is removed, etc), and you might want to keep track of that. So you might want to create a custom type for that:

```python
from pydantic import Basemodel

class Sentence(Basemodel)
    text: str

    def clean(self):
        self.text = clean_text(self.text)

    def raw(self):
        return self.text

class SentenceList(Basemodel):
    sentences: List[Sentence]
```

# Protocol

It is also possible to create your own generic types with `Protocol`, which was introduced
with [PEP 544](https://www.python.org/dev/peps/pep-0544/)
A simple example is this:

```python
from typing import Protocol, Callable, List
from torch import Tensor


class GenericModel(Protocol):
    def train(self, x: Tensor, y: Tensor) -> None:
        ...

    def predict(self, x: Tensor) -> Tensor:
        ...


def train_model(estimators: List[GenericModel]) -> List[GenericModel]:
    ...
```

It doesnt matter how the model is implemented exactly; However, you have some minimal expectations about the methods and the arguments they receive.

Read more about it in [this blog](https://www.daan.fyi/writings/python-protocols)
