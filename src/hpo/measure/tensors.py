"""The shapes of what goes into a model and comes out.

A function that states one of these in its signature, under
`@jaxtyped(typechecker=beartype)`, has it checked on every call: a batch of labels that
is one-hot where indices were expected fails there, by name, instead of three calls on.
"""

from jaxtyping import Float, Int
from numpy import ndarray
from torch import Tensor
from vectormesh.types import TensorInput

# a class index per row
Labels = Int[Tensor, " batch"]
# a score per row and class
Logits = Float[Tensor, "batch classes"]
# what a loader gives as targets: class indices, or one-hot rows
Targets = Int[Tensor, " batch"] | Float[Tensor, "batch classes"]
# one batch of a loader: the model's input and its targets
Batch = tuple[TensorInput, Targets]
# one value of a metric per repeat of a run
PerRepeat = Float[ndarray, " repeat"]
# row = the true class, column = the predicted class
Counts = Int[ndarray, "classes classes"]

__all__ = ["Batch", "Counts", "Labels", "Logits", "PerRepeat", "Targets", "TensorInput"]
