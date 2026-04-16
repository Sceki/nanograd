import math
import random

from .engine import Value


def _as_value(x):
    return x if isinstance(x, Value) else Value(float(x))


def add(a, b):
    a = _as_value(a)
    b = _as_value(b)
    out = Value(a.data + b.data, (a, b), '+')

    def _backward():
        a.grad += out.grad
        b.grad += out.grad

    out._backward = _backward
    return out


def mul(a, b):
    a = _as_value(a)
    b = _as_value(b)
    out = Value(a.data * b.data, (a, b), '*')

    def _backward():
        a.grad += b.data * out.grad
        b.grad += a.data * out.grad

    out._backward = _backward
    return out


def tanh(x):
    x = _as_value(x)
    t = math.tanh(x.data)
    out = Value(t, (x,), 'tanh')

    def _backward():
        x.grad += (1 - t ** 2) * out.grad

    out._backward = _backward
    return out


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        if len(x) != len(self.w):
            raise ValueError(f"Expected {len(self.w)} inputs, got {len(x)}")

        act = self.b
        for wi, xi in zip(self.w, x):
            act = add(act, mul(wi, _as_value(xi)))

        return tanh(act) if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        act_name = 'tanh' if self.nonlin else 'linear'
        return f"Neuron(nin={len(self.w)}, activation={act_name})"


class Layer(Module):
    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer({', '.join(str(n) for n in self.neurons)})"


class MLP(Module):
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            if isinstance(x, Value):
                x = [x]

        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP({', '.join(str(layer) for layer in self.layers)})"
