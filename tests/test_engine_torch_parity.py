import math
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanograd.engine import Value

try:
    import torch  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - optional dependency
    torch = None

def _eml_torch(torch_mod, a, b):
    b = torch_mod.as_tensor(b, dtype=a.dtype, device=a.device)
    return torch_mod.exp(a) - torch_mod.log(b)


def _build_graph_case_one_value(x):
    a = Value.eml(x, 2.5)
    b = Value.eml(a, 1.3)
    c = Value.eml(x, 4.0)
    y = Value.eml(b, c)
    return y


def _build_graph_case_one_torch(torch_mod, x):
    a = _eml_torch(torch_mod, x, 2.5)
    b = _eml_torch(torch_mod, a, 1.3)
    c = _eml_torch(torch_mod, x, 4.0)
    y = _eml_torch(torch_mod, b, c)
    return y


def _build_graph_case_two_value(x, y):
    a = Value.eml(x, y)
    b = Value.eml(y, 4.0)

    c = Value.eml(a, 3.5)
    d = Value.eml(b, a)

    e = Value.eml(c, Value.eml(x, 2.0))
    f = Value.eml(d, Value.eml(y, 5.0))

    g = Value.eml(e, f)
    return g


def _build_graph_case_two_torch(torch_mod, x, y):
    a = _eml_torch(torch_mod, x, y)
    b = _eml_torch(torch_mod, y, 4.0)

    c = _eml_torch(torch_mod, a, 3.5)
    d = _eml_torch(torch_mod, b, a)

    e = _eml_torch(torch_mod, c, _eml_torch(torch_mod, x, 2.0))
    f = _eml_torch(torch_mod, d, _eml_torch(torch_mod, y, 5.0))

    g = _eml_torch(torch_mod, e, f)
    return g


def test_eml_graph_single_variable_parity_with_torch():
    if torch is None:
        return

    xmg = Value(0.7)
    ymg = _build_graph_case_one_value(xmg)
    ymg.backward()

    xpt = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
    ypt = _build_graph_case_one_torch(torch, xpt)
    ypt.backward()

    assert math.isclose(ymg.data, ypt.item(), rel_tol=1e-8, abs_tol=1e-8)
    assert math.isclose(xmg.grad, xpt.grad.item(), rel_tol=1e-8, abs_tol=1e-8)


def test_eml_graph_two_variable_parity_with_torch():
    if torch is None:
        return

    xmg = Value(0.4)
    ymg = Value(1.1)
    gmg = _build_graph_case_two_value(xmg, ymg)
    gmg.backward()

    xpt = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
    ypt = torch.tensor([1.1], dtype=torch.float64, requires_grad=True)
    gpt = _build_graph_case_two_torch(torch, xpt, ypt)
    gpt.backward()

    assert math.isclose(gmg.data, gpt.item(), rel_tol=1e-7, abs_tol=1e-7)
    assert math.isclose(xmg.grad, xpt.grad.item(), rel_tol=1e-7, abs_tol=1e-7)
    assert math.isclose(ymg.grad, ypt.grad.item(), rel_tol=1e-7, abs_tol=1e-7)
