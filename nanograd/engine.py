import cmath
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
    @staticmethod
    def eml(a, b):
        a = a if isinstance(a, Value) else Value(a)
        b = b if isinstance(b, Value) else Value(b)

        def _collapse_real(z):
            if isinstance(z, complex) and abs(z.imag) < 1e-12:
                return float(z.real)
            return z

        def _safe_exp(x):
            try:
                return cmath.exp(complex(x))
            except OverflowError:
                return complex(float('inf'), 0.0)

        def _safe_log(y):
            # Extended-real convention used by EML chains: log(0) = -inf.
            yc = complex(y)
            if yc == 0.0:
                return complex(float('-inf'), 0.0)
            return cmath.log(yc)

        # forward pass
        out = Value(_collapse_real(_safe_exp(a.data) - _safe_log(b.data)), (a, b), 'eml')

        # backward pass
        def _backward():
            # d/da: exp(a)
            a.grad += _collapse_real(_safe_exp(a.data) * out.grad)
            # d/db: -1/b
            if complex(b.data) == 0.0:
                b.grad += _collapse_real(complex(float('-inf'), 0.0) * out.grad)
            else:
                b.grad += _collapse_real((-1.0 / complex(b.data)) * out.grad)

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
