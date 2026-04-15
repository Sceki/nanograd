import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # for debugging

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # --- the ONLY primitive ---
    def eml(self, other):
        other = other if isinstance(other, Value) else Value(other)

        # forward pass
        out = Value(math.exp(self.data) - math.log(other.data), (self, other), 'eml')

        # backward pass
        def _backward():
            # d/dx: exp(x)
            self.grad += math.exp(self.data) * out.grad
            # d/dy: -1/y
            other.grad += (-1.0 / other.data) * out.grad

        out._backward = _backward
        return out

    # --- backprop ---
    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
