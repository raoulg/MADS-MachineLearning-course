## loguru

Often, you want to show some output of your code halfway a process.
While you could use `print` statements, using a logger is a better practice.
You get control over levels of statements, there is more syntax (eg timestamps) to the
log, you can manage logging to files etc.
see the [loguru docs](https://github.com/Delgan/loguru) for more.

Some examples:

```python
from loguru import logger

logger.info("That's it!")
```

By default this will add coloring, timestamps, location etc to your `stderr`
Adding logging files is easy configurable:

```python
logger.add("logfile.log", rotation="1 week")
```

Or decorators to catch exceptions:

```python
@logger.catch
def my_fun(x, y, z):
    # error? it's caught anyway
    return (x + y) / z
```
