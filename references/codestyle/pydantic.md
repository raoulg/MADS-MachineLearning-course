# Pydantic

While creating larger projects, we will typically end up having a lot of parameters. While the fastest way might seem to just hardcode them somewhere, this is not a valid long-term strategy.

Especially when doing experiments with machine learning, we will want to have everything in one place, and ideally we want to have checks in place.

## Creating configuration settings

To start naively, we could just make a config like this

```python
config = {"input_size": 3, "output_size": 20, "data_dir": "."}
```

While this will work perfectly fine if you are prototyping, at some point your project will grow bigger and you will want to have more control over your parameters.

Pydantic helps with a lot of things. First of all, it will help you to define the types of your parameters. This will help you to catch bugs early on!
At the moment the wrong type is passed, pydantic will first try to convert it to the correct type. If this fails, it will raise an error.

```python
class SearchSpace(BaseModel):
    input_size: int
    output_size: int
    tune_dir: Optional[Path]
    data_dir: Path


config = SearchSpace(input_size=3.0, output_size=20, data_dir=".")
```

In this code, the inputsize is provided as a float but will be casted to an int. The data_dir is provided as a string, but will be casted to a Path object. The ommited tune_dir will be set to None. Python will happily continue if you provide the wrong type; it will even do calculations with strings, etc, potentially making it difficult to debug errors in larger codebases. In addition to that, it helps to read the code and to clarify what type of parameter is expected.

## root validators

It is possible to extend the settings with a `@root_validator`. This will allow you to do some checks on the parameters.

```python
class SearchSpace(BaseModel):
    input_size: int
    output_size: int
    tune_dir: Optional[Path]
    data_dir: Path

    @root_validator
    def check_path(cls, values: Dict) -> Dict:
        datadir = values.get("data_dir")
        if not datadir.exists():
            raise FileNotFoundError(
                f"Make sure the datadir exists.\n Found {datadir} to be non-existing."
            )
        return values
```

In this example, the location of the data_dir is checked. This is a really common cause of errors! You could also choose to create the directory if it does not exist yet.

## Mutable attributes

Another issue that should give every python programmer nightmares, is this:

```python
class MyClass:
    mutable_attr = []


# Create two instances
instance1 = MyClass()
instance2 = MyClass()

# Append to the list in instance1
instance1.mutable_attr.append("Hello")

print(instance1.mutable_attr)  # prints ['Hello']
print(instance2.mutable_attr)  # also prints ['Hello']. Wait, what?
```

This is because mutable attributes are shared between instances. This is a common source of bugs, and can be prevented by using pydantic.

```python
from pydantic import BaseModel
from typing import List


class TrainerSettings(BaseModel):
    mutable_attr: List = []


# Create two settings instances
settings1 = TrainerSettings()
settings2 = TrainerSettings()

# Change 'factor' in settings1
settings1.mutable_attr.append("Hello")

print(settings1.mutable_attr)  # prints ["Hello"]
print(settings2.mutable_attr)  # print []
```
