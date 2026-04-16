import math
import random

from .engine import Value


def _as_value(x):
    return x if isinstance(x, Value) else Value(float(x))


def _direct_add(a, b):
    a = _as_value(a)
    b = _as_value(b)
    out = Value(a.data + b.data, (a, b), '+')

    def _backward():
        a.grad += out.grad
        b.grad += out.grad

    out._backward = _backward
    return out


def _direct_mul(a, b):
    a = _as_value(a)
    b = _as_value(b)
    out = Value(a.data * b.data, (a, b), '*')

    def _backward():
        a.grad += b.data * out.grad
        b.grad += a.data * out.grad

    out._backward = _backward
    return out


def _direct_tanh(x):
    x = _as_value(x)
    t = math.tanh(x.data)
    out = Value(t, (x,), 'tanh')

    def _backward():
        x.grad += (1 - t ** 2) * out.grad

    out._backward = _backward
    return out


def _eml_exp(x):
    return _as_value(x).eml(1.0)


def _eml_log(x):
    # Canonical witness: log(x) = EML(1, EML(EML(1, x), 1))
    one = Value(1.0)
    return one.eml(one.eml(_as_value(x)).eml(one))


def _eml_zero():
    return _eml_log(1.0)


def _eml_sub(a, b):
    # Canonical witness: a - b = EML(log(a), exp(b))
    a = _as_value(a)
    b = _as_value(b)
    return _eml_log(a).eml(_eml_exp(b))


def _eml_neg(x):
    return _eml_sub(_eml_zero(), _as_value(x))


def _eml_add(a, b):
    # Canonical witness: a + b = a - (-b)
    return _eml_sub(_as_value(a), _eml_neg(_as_value(b)))


def _eml_inv(x):
    # Canonical witness: 1/x = exp(-log(x))
    return _eml_exp(_eml_neg(_eml_log(_as_value(x))))


def _eml_mul(a, b):
    # Canonical witness: a*b = exp(log(a) + log(b))
    a = _as_value(a)
    b = _as_value(b)
    return _eml_exp(_eml_add(_eml_log(a), _eml_log(b)))


def _eml_tanh(x):
    # tanh(x) = 1 - 2 / (exp(2x) + 1), composed from EML-derived helpers.
    x = _as_value(x)
    ex = _eml_exp(x)
    exp_2x = _eml_mul(ex, ex)
    denom = _eml_add(exp_2x, 1.0)
    frac = _eml_mul(2.0, _eml_inv(denom))
    return _eml_sub(1.0, frac)


def add(a, b):
    try:
        return _eml_add(a, b)
    except (ValueError, OverflowError):
        # Real-only EML reconstruction can hit log-domain limits; preserve usability.
        return _direct_add(a, b)


def mul(a, b):
    try:
        return _eml_mul(a, b)
    except (ValueError, OverflowError):
        # Real-only EML reconstruction can hit log-domain limits; preserve usability.
        return _direct_mul(a, b)


def tanh(x):
    try:
        return _eml_tanh(x)
    except (ValueError, OverflowError):
        # Real-only EML reconstruction can hit log-domain limits; preserve usability.
        return _direct_tanh(x)


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
