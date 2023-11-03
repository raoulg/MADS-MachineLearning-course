# Linting

By default, I will install the following libraries for linting:

- `black`
- `ruff`
- `isort`
- `mypy`

And I will create a `Makefile` with this template:

```makefile
.PHONY: lint format

format:
        isort -v src
        black src

lint:
        ruff src
        mypy src
```

This will allow you to run `make format` from the terminal to format your code, and `make lint` to lint your code.
Alternatively, you can also run directly from the command line things like `black src`,
which will format all python files inside the `src/` folder

# Formatters

`black` calls itself the `uncompromising code formatter`. It is very opiniated about
formatting code, but does this in a consistent style. You hand over control about
formatting, but you will get consistency in return and never have to think about formatting
again. read the [black docs](https://github.com/psf/black)

`isort` is much less invasive. Only thing it does is sorting your imports. But, yeah, why
not, it is anoying to sort those too so why not automate that [isort](https://pycqa.github.io/isort)

# Type checkers

After running the formatters, I run [ruff](https://beta.ruff.rs/docs/)

It is fast, supports the `pyproject.toml` standard and has sane defaults.

While black will already take care of all the nitty gritty PEP 8 details, there will still be some
issues that ruff will catch.

However, some checks are too tight or conflicting (eg black limits line length to 88 (or some other standard you pick), so ruff needs to know about that), which is where the `pyproject.toml` configuration file comes in.
Have a look at the pyproject.toml file in this repo for an example, check the `tool.ruff` and `tool.black` sections.

After you added proper typehinting everywhere, `mypy` will start checking consistency.

Imagine you have a function that retrieves the median of a list of customers,
and it is important that the customers are discrete counts, and thus an integer.
The custom median function will return an `int`, but the `get_value` function expects a `float`.

```python
def median(values: List[int]) -> int
    ...
    return x

def get_value(data: List[float]) -> float
    return median(data) # inconsistent: this returns an int
```

This is inconsistent, and you should make up your mind about the types.
Obviously, this is a simple example, but at some point you might get pretty complex datastructures and mypy will help you to keep your code consistent and warn you about loopholes that might cause errors, even though your code would function fine when you are testing it now.

While it might seem like a new source of frustration, mypy will help you to

- write better code
- spend less time debugging later on (in production)
- understand your code better
- make your code more readable

Which are more than enough reasons to use it.
