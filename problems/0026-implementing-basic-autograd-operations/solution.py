
import torch

class Value:
    """A tiny scalar wrapper that delegates gradient work to PyTorch autograd."""

    def __init__(self, data, _tensor=None):
        self._t = (
            _tensor if _tensor is not None
            else torch.tensor(float(data), requires_grad=True)
        )

        # Retain gradients for intermediate tensors too
        if self._t.requires_grad:
            self._t.retain_grad()

    # Properties
    @property
    def data(self):
        return self._t.item()

    @property
    def grad(self):
        g = self._t.grad
        return 0 if g is None else g.item()

    def __repr__(self):
        def fmt(x):
            return int(x) if float(x).is_integer() else round(x, 4)
        return f"Value(data={fmt(self.data)}, grad={fmt(self.grad)})"

    # Convert normal numbers into Value objects
    def _wrap(self, other):
        return other if isinstance(other, Value) else Value(other)

    # Addition
    def __add__(self, other):
        other = self._wrap(other)
        result = self._t + other._t
        return Value(None, _tensor=result)

    __radd__ = __add__

    # Multiplication
    def __mul__(self, other):
        other = self._wrap(other)
        result = self._t * other._t
        return Value(None, _tensor=result)

    __rmul__ = __mul__

    # ReLU
    def relu(self):
        result = torch.relu(self._t)
        return Value(None, _tensor=result)

    # Backward
    def backward(self):
        self._t.backward()
